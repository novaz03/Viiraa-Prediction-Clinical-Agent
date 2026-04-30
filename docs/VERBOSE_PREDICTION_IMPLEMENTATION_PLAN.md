# Viiraa Webapp: Verbose Prediction & Advanced Analysis Implementation Plan

## Goal
Upgrade the web app so predictions are more prominent, clinically interpretable (non-diagnostic), and better contextualized using cohort and personal meal comparisons.

## TODO (Product Direction Updates)
1. Consolidate to one user-facing input mode only:
2. User provides meal + demographics + labs + pre-meal glucose series.
3. Backend computes all engineered features automatically.
4. Deprecate/hide separate engineered-feature entry mode from UI and external docs.
5. Add persistent profile storage for demographics and labs:
6. Implement a database-backed demographics/lab profile store (user-linked) so users do not re-enter static profile fields each time.

---

## Requirements (From User)

1. Highlight predictions and hide precalculable premeal-glucose-curve features.
2. Improve visualization for `peak_amplitude`, `iauc_120`, and `auc_120_abs`.
3. Add histograms (~30 bins) showing where a meal’s predicted value lies among same meal-type meals, with the active bin highlighted.
4. Add confidence intervals for predictions.
5. Incorporate demographics (age, height, weight, and other model-required demographic features).
6. Remove `step_minutes` as user input; keep fixed internally.
7. Improve verbal analysis:
8. Short-term impact framing (spike severity, potential danger language with safeguards).
9. Compare against user’s previous meals and “typical meals” from CGMacros.
10. Provide API example after implementation.

---

## Architecture Changes

1. Keep current prediction API and extend response with:
2. Prediction intervals
3. Cohort histogram metadata
4. Cohort percentile metadata
5. Personal-history comparison metadata
6. Narrative analysis text blocks
7. Add a reference distribution component:
8. Precompute 30-bin histograms by `meal_type` and target metric.
9. Store distribution artifacts in versioned static data files.
10. Add a user-history store:
11. Session/user keyed historical predictions.
12. Start with local persisted store (JSONL/Parquet), with migration path to DB.

---

## Backend Implementation Plan

## Phase A: Input Contract and Feature Coverage
1. Audit model-required columns using checkpoint feature metadata (`expected_columns`).
2. Define demographic fields that are required/optional with validation ranges.
3. Update Pydantic schemas:
4. Add demographic fields.
5. Remove `step_minutes` from frontend-facing contract.
6. Keep fixed defaults server-side for raw feature derivation.
7. Add schema docs table (`field`, `type`, `required`, `source`, `validation`).

## Phase B: Confidence Intervals
1. Use fold-model ensemble (`best_fold_model`) to estimate prediction spread.
2. Calibrate uncertainty using residual statistics (global + meal-type fallback).
3. Return:
4. `ci_80` and `ci_95` per target.
5. Method metadata (`ensemble_calibrated_v1`, sample count, calibration set id).
6. Add tests for interval generation and monotonic consistency (`ci_95` wider than `ci_80`).

## Phase C: Cohort Distribution and Histograms
1. Build preprocessing script to create per-meal-type histograms for each metric:
2. 30 bins for `auc_120_abs`
3. 30 bins for `iauc_120`
4. 30 bins for `peak_amplitude`
5. Save artifact format:
6. Bin edges
7. Bin counts
8. Percentile lookup support
9. At inference response time, compute:
10. Predicted bin index
11. Percentile vs same meal type
12. Cohort comparison descriptors (low/mid/high tail)

## Phase D: Personal Meal Comparison
1. Persist user predictions with minimal profile + timestamp + meal type.
2. Compute comparisons:
3. Delta vs previous meal of same type
4. Delta vs rolling median (last N meals)
5. Percentile within user history (if enough records)
6. Add confidence/availability flags when insufficient history.

## Phase E: Verbal Analysis Engine
1. Implement rule-based narrative module with guardrails:
2. Short-term impact phrasing based on predicted magnitude and CI overlap.
3. “Potential spike concern” language with strict non-diagnostic disclaimer.
4. Cohort comparison narrative (“higher/lower than typical [meal_type] meals”).
5. Personal comparison narrative (“higher/lower than your recent [meal_type] meals”).
6. Return structured text sections:
7. `headline`
8. `short_term_impact`
9. `cohort_comparison`
10. `personal_comparison`
11. `safety_note`

---

## Frontend Implementation Plan

1. Redesign result area so prediction cards are first and most prominent.
2. Move derived/precalculated premeal feature inputs into hidden “Advanced” accordion (collapsed by default).
3. Remove `step_minutes` input entirely from UI.
4. Add dedicated chart blocks (one per target):
5. 30-bin histogram bars
6. Highlight predicted-value bin
7. Display percentile and CI labels near chart title
8. Add summary comparison cards:
9. Typical meal-type comparison
10. Personal recent-meal comparison
11. Add narrative section with structured analysis text from backend.

---

## API Changes (Planned)

## Request
1. Standardize on one user-facing request mode (`raw_input`-style payload with meal + demographics + labs + pre-glucose series).
2. Add demographics fields to accepted input (required/optional per schema).
3. Keep raw feature derivation internally fixed for sampling settings (no `step_minutes` input).
4. Keep engineered-feature mode only as temporary internal/backward-compatibility path until removal.

## Response Additions
1. `prediction_intervals`
2. `cohort_comparison`
3. `personal_comparison`
4. `analysis_text`
5. `artifact_versions` (model/distribution/narrative rule versions)

---

## API Example (To Ship with Implementation)

After implementation, include a concrete example in docs and `/v1/example-input` companion output.

Example response shape (illustrative):

```json
{
  "request_id": "9b5c4f38-...",
  "predictions": {
    "auc_120_abs": 11854.2,
    "iauc_120": 1760.4,
    "peak_amplitude": 43.7
  },
  "prediction_intervals": {
    "auc_120_abs": { "ci_80": [10320.1, 13240.8], "ci_95": [9580.2, 14110.6] },
    "iauc_120": { "ci_80": [1402.3, 2054.7], "ci_95": [1260.5, 2238.1] },
    "peak_amplitude": { "ci_80": [36.1, 50.4], "ci_95": [32.5, 54.8] }
  },
  "cohort_comparison": {
    "meal_type": "Lunch",
    "auc_120_abs": { "percentile": 62.1, "bin_index": 18, "bin_count": 30 },
    "iauc_120": { "percentile": 58.4, "bin_index": 17, "bin_count": 30 },
    "peak_amplitude": { "percentile": 64.9, "bin_index": 19, "bin_count": 30 }
  },
  "personal_comparison": {
    "history_count": 24,
    "vs_recent_same_meal_type": {
      "auc_120_abs_delta": 520.6,
      "iauc_120_delta": 84.2,
      "peak_amplitude_delta": 3.1
    }
  },
  "analysis_text": {
    "headline": "Predicted excursion is moderately above your recent Lunch pattern.",
    "short_term_impact": "This meal profile suggests a noticeable post-meal glucose rise over the next 2 hours.",
    "cohort_comparison": "Compared with typical Lunch meals in reference data, this result is slightly higher.",
    "personal_comparison": "Compared with your recent Lunch meals, excursion is moderately higher.",
    "safety_note": "For research use only; not diagnostic or treatment advice."
  },
  "artifact_versions": {
    "model_family": "scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget",
    "ci_method": "ensemble_calibrated_v1",
    "cohort_reference": "cgmacros_hist_v1",
    "analysis_rules": "verbal_rules_v1"
  }
}
```

---

## Testing & Validation

1. Contract tests for new schema and response keys.
2. Histogram logic tests:
3. Bin assignment correctness
4. Highlight index correctness
5. Percentile monotonic behavior
6. CI tests:
7. Ordering and width checks
8. Non-empty interval constraints
9. Narrative tests:
10. Expected text for low/moderate/high scenarios
11. Guardrail disclaimer always present
12. End-to-end UI tests for desktop and mobile.

---

## Rollout Plan

1. Backend schema and inference metadata (demographics + CI).
2. Cohort histogram artifacts + API integration.
3. Personal history comparison.
4. Narrative engine.
5. Frontend redesign and charts.
6. Docs and API example finalization.
7. QA and deployment.
