# Live Per-Meal CI Computation Brief

## Goal
Design a robust approach to compute **prediction intervals live for each meal request** and support reliable summary statistics (for example, average 90% interval width over CGMacros meals).

## Current Situation
- The API currently serves per-request intervals using the deployed model stack.
- Existing “average 90% CI width” numbers were derived from offline OOF artifacts (`oof_predictions.parquet`), not fully recomputed live per request.
- This creates a mismatch between:
  - offline summary metrics, and
  - online interval method and runtime behavior.

## Problem to Solve
We need a system where:
1. Per-meal interval computation is available live at inference time.
2. Aggregated interval-width statistics are consistent with the same interval method.
3. Outputs are traceable by model version, interval method, and calibration version.

---

## Required Components

### 1) Interval Method (Per-meal uncertainty engine)
Pick one method and make it explicit in metadata.

Options:
- **Current interim**: fold-sigma spread + residual widening.
- **Preferred long-term**: quantile model + conformal calibration.

Requirement:
- Return method id in every response (for example: `fold_sigma_plus_resid_v1`, `quantile_conformal_v1`).

### 2) Live Per-Meal Interval Endpoint Logic
For each incoming meal:
1. Build/validate features.
2. Produce point prediction.
3. Produce interval bounds per target (including 90% if needed).
4. Return interval width per target:
   - `width_90 = hi_90 - lo_90`

Suggested response fields:
- `predictions`
- `intervals` (for example `interval_90`, `interval_95`)
- `interval_widths` (for example `width_90`)
- `interval_method`
- `model_id`, `model_family`, `calibration_version`

### 3) Reference Aggregation Layer (CGMacros summaries)
Need consistent aggregation for questions like:
- “What is the average 90% interval width for all CGMacros meals?”

Two implementation modes:
- **Precomputed cache** (recommended):
  - Batch compute interval widths across all reference meals.
  - Store compact table by target + meal type + model/method version.
- **On-demand compute** (slower):
  - Recompute widths over all meals per query.

Recommended summary outputs:
- mean width
- median width
- percentile bands (p10/p90)
- sample size (`n`)
- grouping (overall, meal type, user subset)
- method/model/calibration/version stamp

---

## Proposed API Additions

### A) Live single-meal interval endpoint
- `POST /v1/interval-width` (or extend `/v1/predict`)

Returns:
- per-target interval bounds and widths
- interval method metadata

### B) Reference summary endpoint
- `GET /v1/reference-interval-widths`

Query filters:
- `target`
- `meal_type` (optional)
- `model_id`
- `interval_method`

Returns:
- aggregate stats and provenance metadata

---

## Data/Versioning Requirements
- Track and return:
  - model artifact id/path hash
  - interval method id
  - calibration artifact id/version
  - data snapshot id for CGMacros reference
- Store computation timestamp for cached summaries.

---

## Acceptance Criteria
1. Live endpoint returns stable per-meal interval width for each target.
2. Aggregate endpoint reports reproducible stats for the same model+method.
3. API metadata is sufficient to audit “how this width was computed.”
4. No mismatch between UI-reported method and backend method.
5. Runtime latency remains acceptable.

---

## Open Design Questions (for AI solver)
1. Should 90% intervals be first-class in API, or derived from another representation?
2. Should interval computation live inside model service or a separate uncertainty service?
3. What caching strategy best balances freshness vs latency for CGMacros summaries?
4. How should calibration be versioned and rolled back safely?
5. How to migrate from current fold-sigma method to quantile+conformal without breaking API consumers?

---

## Recommended Near-Term Plan
1. Add `interval_90` + `width_90` to live prediction responses.
2. Build a nightly/precompute job for CGMacros interval-width summaries.
3. Expose summary endpoint with strict model/method version tags.
4. Keep API schema stable while swapping interval engine later.
5. Evaluate and migrate to quantile+conformal once validated.

