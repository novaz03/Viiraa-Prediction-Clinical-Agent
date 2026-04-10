from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


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


def build_required_features_from_raw(
    meal_info: Dict[str, Any],
    pre_glucose_series: List[Any],
    step_minutes: float = 5.0,
    baseline_window_minutes: float = 30.0,
) -> Dict[str, Any]:
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

    out = {
        "meal_type": str(meal_info.get("meal_type", "Lunch")),
        "meal_calories": float(meal_info.get("meal_calories", 0.0)),
        "carbs_g": float(meal_info.get("carbs_g", 0.0)),
        "protein_g": float(meal_info.get("protein_g", 0.0)),
        "fat_g": float(meal_info.get("fat_g", 0.0)),
        "A1c PDL (Lab)": float(meal_info.get("A1c PDL (Lab)", meal_info.get("a1c_lab", 0.0))),
        "Fasting GLU - PDL (Lab)": float(
            meal_info.get("Fasting GLU - PDL (Lab)", meal_info.get("fasting_glu_lab", 0.0))
        ),
        "minutes_since_last_meal": float(meal_info.get("minutes_since_last_meal", 0.0)),
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
        "pre_glucose_valid_count": float(valid_vals.size),
        "pre_glucose_longest_gap": float(longest_gap),
        "premeal_baseline_glucose": baseline_mean_30m,
        "Age": float(meal_info.get("Age", meal_info.get("age", 0.0))),
        "Gender": str(meal_info.get("Gender", meal_info.get("gender", meal_info.get("sex", "")))),
        "Height": float(meal_info.get("Height", meal_info.get("height", 0.0))),
        "BMI": float(meal_info.get("BMI", meal_info.get("bmi", 0.0))),
        "Body weight": float(
            meal_info.get(
                "Body weight",
                meal_info.get("body_weight", meal_info.get("weight", meal_info.get("Weight", 0.0))),
            )
        ),
    }
    return out
