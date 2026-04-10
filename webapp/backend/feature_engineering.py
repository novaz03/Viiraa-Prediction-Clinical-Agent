from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

REQUIRED_MEAL_INFO_ALIASES: Dict[str, List[str]] = {
    "meal_type": ["meal_type"],
    "meal_calories": ["meal_calories"],
    "carbs_g": ["carbs_g"],
    "protein_g": ["protein_g"],
    "fat_g": ["fat_g"],
    "minutes_since_last_meal": ["minutes_since_last_meal"],
    "A1c PDL (Lab)": ["A1c PDL (Lab)", "a1c_lab"],
    "Fasting GLU - PDL (Lab)": ["Fasting GLU - PDL (Lab)", "fasting_glu_lab"],
    "Age": ["Age", "age"],
    "Gender": ["Gender", "gender", "sex", "Sex"],
    "Height": ["Height", "height"],
    "BMI": ["BMI", "bmi"],
    "Body weight": ["Body weight", "body_weight", "weight", "Weight"],
}


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _linear_slope(values: np.ndarray, step_minutes: float) -> float:
    n = int(values.size)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64) * float(step_minutes)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(values))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-12:
        return 0.0
    num = float(np.sum((x - x_mean) * (values - y_mean)))
    return num / den


def _longest_missing_gap_minutes(raw: List[Any], step_minutes: float) -> float:
    longest = 0
    cur = 0
    for v in raw:
        if _safe_float(v) is None:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return float(longest) * float(step_minutes)


def _require_raw_value(meal_info: Dict[str, Any], canonical: str) -> Any:
    for key in REQUIRED_MEAL_INFO_ALIASES.get(canonical, [canonical]):
        if key in meal_info and meal_info[key] not in (None, ""):
            return meal_info[key]
    raise ValueError(f"Missing required raw field in meal_info: `{canonical}`")


def _require_raw_numeric(meal_info: Dict[str, Any], canonical: str, *, min_value: float | None = None) -> float:
    val = _require_raw_value(meal_info, canonical)
    try:
        out = float(val)
    except Exception as exc:
        raise ValueError(f"Raw field `{canonical}` must be numeric.") from exc
    if not np.isfinite(out):
        raise ValueError(f"Raw field `{canonical}` must be finite.")
    if min_value is not None and out < min_value:
        raise ValueError(f"Raw field `{canonical}` must be >= {min_value}.")
    return out


def _require_raw_text(meal_info: Dict[str, Any], canonical: str) -> str:
    val = _require_raw_value(meal_info, canonical)
    out = str(val).strip()
    if not out:
        raise ValueError(f"Raw field `{canonical}` must be non-empty.")
    return out


def build_required_features_from_raw(
    meal_info: Dict[str, Any],
    pre_glucose_series: List[Any],
    step_minutes: float = 5.0,
    baseline_window_minutes: float = 30.0,
) -> Dict[str, Any]:
    if not isinstance(meal_info, dict):
        raise ValueError("meal_info must be a JSON object.")

    if not isinstance(pre_glucose_series, list) or len(pre_glucose_series) < 4:
        raise ValueError("pre_glucose_series must be a list with at least 4 values.")

    if step_minutes <= 0:
        raise ValueError("step_minutes must be > 0.")
    if baseline_window_minutes <= 0:
        raise ValueError("baseline_window_minutes must be > 0.")

    raw = list(pre_glucose_series)
    vals = np.array([_safe_float(v) for v in raw], dtype=object)
    valid_mask = np.array([v is not None for v in vals], dtype=bool)
    valid_vals = np.array([float(v) for v in vals[valid_mask]], dtype=np.float64)
    if valid_vals.size < 4:
        raise ValueError("Need at least 4 finite glucose values in pre_glucose_series.")

    n = len(raw)
    missing_frac = 1.0 - float(valid_vals.size) / float(n)
    longest_gap = _longest_missing_gap_minutes(raw, step_minutes=step_minutes)

    # Fill missing by linear interpolation for slope calculations.
    interp = np.array(
        [np.nan if _safe_float(v) is None else float(v) for v in raw],
        dtype=np.float64,
    )
    idx = np.arange(interp.size, dtype=np.float64)
    good = np.isfinite(interp)
    if not np.all(good):
        interp[~good] = np.interp(idx[~good], idx[good], interp[good])

    baseline_n = max(1, int(round(baseline_window_minutes / step_minutes)))
    baseline_vals = interp[-baseline_n:]
    baseline_mean_30m = float(np.mean(baseline_vals))
    baseline_median_30m = float(np.median(baseline_vals))

    # Segment windows from the end:
    # - 180..60 min segment: exclude last 12 points (for 5-min data), take earlier portion
    # - 60..15 min: preceding 9 points
    # - 15..0 min: last 3 points
    pts_60 = max(1, int(round(60.0 / step_minutes)))
    pts_15 = max(1, int(round(15.0 / step_minutes)))
    seg_180_60 = interp[: max(2, interp.size - pts_60)]
    seg_60_15 = interp[max(0, interp.size - pts_60) : max(2, interp.size - pts_15)]
    seg_15_0 = interp[max(0, interp.size - pts_15) :]

    slope_180_60 = float(_linear_slope(seg_180_60, step_minutes=step_minutes))
    slope_60_15 = float(_linear_slope(seg_60_15, step_minutes=step_minutes))
    slope_15_0 = float(_linear_slope(seg_15_0, step_minutes=step_minutes))
    slope_recent_minus_early = slope_15_0 - slope_180_60

    mn = float(np.min(valid_vals))
    mx = float(np.max(valid_vals))
    std = float(np.std(valid_vals))
    mean_all = float(np.mean(valid_vals))
    cv = std / max(abs(mean_all), 1e-6)
    iqr = float(np.percentile(valid_vals, 75) - np.percentile(valid_vals, 25))

    meal_type = _require_raw_text(meal_info, "meal_type")
    meal_calories = _require_raw_numeric(meal_info, "meal_calories", min_value=0.0)
    carbs_g = _require_raw_numeric(meal_info, "carbs_g", min_value=0.0)
    protein_g = _require_raw_numeric(meal_info, "protein_g", min_value=0.0)
    fat_g = _require_raw_numeric(meal_info, "fat_g", min_value=0.0)
    minutes_since_last_meal = _require_raw_numeric(meal_info, "minutes_since_last_meal", min_value=0.0)
    a1c_lab = _require_raw_numeric(meal_info, "A1c PDL (Lab)", min_value=0.0)
    fasting_lab = _require_raw_numeric(meal_info, "Fasting GLU - PDL (Lab)", min_value=0.0)
    age = _require_raw_numeric(meal_info, "Age", min_value=0.0)
    gender = _require_raw_text(meal_info, "Gender")
    height = _require_raw_numeric(meal_info, "Height", min_value=0.0)
    # Align with model training convention: Height stored in meters.
    # Users commonly enter centimeters (e.g., 165), so normalize when value > 3.
    if height > 3.0:
        height = height / 100.0
    bmi = _require_raw_numeric(meal_info, "BMI", min_value=0.0)
    body_weight = _require_raw_numeric(meal_info, "Body weight", min_value=0.0)

    out = {
        "meal_type": meal_type,
        "meal_calories": meal_calories,
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "A1c PDL (Lab)": a1c_lab,
        "Fasting GLU - PDL (Lab)": fasting_lab,
        "minutes_since_last_meal": minutes_since_last_meal,
        "baseline_glucose_median_30m": baseline_median_30m,
        "baseline_glucose_mean_30m": baseline_mean_30m,
        "pre_glucose_min_180m": mn,
        "pre_glucose_max_180m": mx,
        "pre_glucose_range_180m": mx - mn,
        "pre_glucose_iqr_180m": iqr,
        "pre_glucose_std_180m": std,
        "pre_glucose_cv_180m": cv,
        "glucose_slope_180_60": slope_180_60,
        "glucose_slope_60_15": slope_60_15,
        "glucose_slope_15_0": slope_15_0,
        "glucose_slope_recent_minus_early": slope_recent_minus_early,
        "pre_glucose_missing_frac": float(missing_frac),
        # Training used minute-scale valid-coverage for this field.
        "pre_glucose_valid_count": float(valid_vals.size) * float(step_minutes),
        "pre_glucose_longest_gap": float(longest_gap),
        "premeal_baseline_glucose": baseline_mean_30m,
        "Age": age,
        "Gender": gender,
        "Height": height,
        "BMI": bmi,
        "Body weight": body_weight,
    }
    return out
