from __future__ import annotations

from typing import Any, Dict

import numpy as np


REQUIRED_FIELDS = [
    "meal_type",
    "meal_calories",
    "carbs_g",
    "protein_g",
    "fat_g",
    "A1c PDL (Lab)",
    "Fasting GLU - PDL (Lab)",
    "minutes_since_last_meal",
    "baseline_glucose_median_30m",
    "baseline_glucose_mean_30m",
    "pre_glucose_min_180m",
    "pre_glucose_max_180m",
    "pre_glucose_range_180m",
    "pre_glucose_iqr_180m",
    "pre_glucose_std_180m",
    "pre_glucose_cv_180m",
    "glucose_slope_180_60",
    "glucose_slope_60_15",
    "glucose_slope_15_0",
    "glucose_slope_recent_minus_early",
    "pre_glucose_missing_frac",
    "pre_glucose_valid_count",
    "pre_glucose_longest_gap",
    "premeal_baseline_glucose",
]
NUMERIC_REQUIRED = [f for f in REQUIRED_FIELDS if f != "meal_type"]
REQUIRED_DEMOGRAPHIC_NUMERIC = ["Age", "BMI", "Height", "Body weight"]
REQUIRED_DEMOGRAPHIC_CATEGORICAL = ["Gender"]
DEMO_ALIASES: Dict[str, list[str]] = {
    "Age": ["Age", "age"],
    "Gender": ["Gender", "gender", "sex", "Sex"],
    "BMI": ["BMI", "bmi"],
    "Height": ["Height", "height"],
    "Body weight": ["Body weight", "body_weight", "weight", "Weight"],
}


def validate_and_normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(features, dict):
        raise ValueError("`features` must be a JSON object.")

    missing = [k for k in REQUIRED_FIELDS if k not in features]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    out = dict(features)
    for canonical, aliases in DEMO_ALIASES.items():
        if canonical in out and out[canonical] not in (None, ""):
            continue
        for alias in aliases:
            if alias in out and out[alias] not in (None, ""):
                out[canonical] = out[alias]
                break

    for col in NUMERIC_REQUIRED:
        try:
            out[col] = float(out[col])
        except Exception as exc:
            raise ValueError(f"Field `{col}` must be numeric.") from exc

    for col in REQUIRED_DEMOGRAPHIC_NUMERIC:
        if col not in out:
            raise ValueError(f"Missing required demographic field: {col}")
        try:
            out[col] = float(out[col])
        except Exception as exc:
            raise ValueError(f"Demographic field `{col}` must be numeric.") from exc
        if not np.isfinite(out[col]) or float(out[col]) <= 0:
            raise ValueError(f"Demographic field `{col}` must be > 0.")

    for col in REQUIRED_DEMOGRAPHIC_CATEGORICAL:
        if col not in out or str(out[col]).strip() == "":
            raise ValueError(f"Missing required demographic field: {col}")
        out[col] = str(out[col]).strip()

    out["meal_type"] = str(out["meal_type"])
    return out

