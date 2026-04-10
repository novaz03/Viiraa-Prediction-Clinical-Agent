from __future__ import annotations

import unittest

from pydantic import ValidationError

from webapp.backend.feature_engineering import build_required_features_from_raw
from webapp.backend.schemas import PredictRequest
from webapp.backend.validation import validate_and_normalize_features


def _sample_raw_input() -> dict:
    return {
        "meal_info": {
            "meal_type": "Lunch",
            "meal_calories": 650,
            "carbs_g": 70,
            "protein_g": 32,
            "fat_g": 22,
            "minutes_since_last_meal": 180,
            "A1c PDL (Lab)": 5.7,
            "Fasting GLU - PDL (Lab)": 95,
            "Age": 42,
            "Gender": "F",
            "Height": 65,
            "BMI": 24.8,
            "Body weight": 149,
        },
        "pre_glucose_series": [92, 93, 94, 95, 96, 97, 98, 99],
    }


class PhaseAContractTests(unittest.TestCase):
    def test_predict_request_requires_raw_input(self) -> None:
        with self.assertRaises(ValidationError):
            PredictRequest()

    def test_predict_request_accepts_raw_input(self) -> None:
        req = PredictRequest(raw_input=_sample_raw_input(), user_id="u1")
        self.assertEqual(req.user_id, "u1")
        self.assertEqual(req.raw_input.meal_info["meal_type"], "Lunch")

    def test_validation_rejects_missing_required_demographic(self) -> None:
        raw = _sample_raw_input()
        raw["meal_info"].pop("Age")
        with self.assertRaisesRegex(ValueError, "Missing required raw field in meal_info: `Age`"):
            build_required_features_from_raw(
                meal_info=raw["meal_info"],
                pre_glucose_series=raw["pre_glucose_series"],
                step_minutes=5.0,
                baseline_window_minutes=30.0,
            )

    def test_validation_rejects_non_positive_demographic_numeric(self) -> None:
        raw = _sample_raw_input()
        raw["meal_info"]["BMI"] = 0
        feats = build_required_features_from_raw(
            meal_info=raw["meal_info"],
            pre_glucose_series=_sample_raw_input()["pre_glucose_series"],
            step_minutes=5.0,
            baseline_window_minutes=30.0,
        )
        with self.assertRaisesRegex(ValueError, "Demographic field `BMI` must be > 0"):
            validate_and_normalize_features(feats)

    def test_build_features_rejects_missing_required_meal_field(self) -> None:
        raw = _sample_raw_input()
        raw["meal_info"].pop("meal_calories")
        with self.assertRaisesRegex(ValueError, "Missing required raw field in meal_info: `meal_calories`"):
            build_required_features_from_raw(
                meal_info=raw["meal_info"],
                pre_glucose_series=raw["pre_glucose_series"],
                step_minutes=5.0,
                baseline_window_minutes=30.0,
            )

    def test_build_features_normalizes_metric_height_weight(self) -> None:
        raw = _sample_raw_input()
        raw["meal_info"]["Height"] = 165
        raw["meal_info"]["Body weight"] = 67.5
        feats = build_required_features_from_raw(
            meal_info=raw["meal_info"],
            pre_glucose_series=raw["pre_glucose_series"],
            step_minutes=5.0,
            baseline_window_minutes=30.0,
        )
        self.assertAlmostEqual(feats["Height"], 64.96062992125984, places=6)
        self.assertAlmostEqual(feats["Body weight"], 148.81202697479486, places=6)


if __name__ == "__main__":
    unittest.main()
