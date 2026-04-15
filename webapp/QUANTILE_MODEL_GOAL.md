# Quantile Model Goal (Future CI Upgrade)

## Objective
Replace current sigma/spread-based uncertainty bands with quantile-model-based prediction intervals that are better aligned with empirical coverage.

## Why
- Current intervals are derived from fold spread + residual widening.
- Fold spread is not a true probabilistic CI.
- We need interval outputs that are more stable, interpretable, and calibratable.

## Target Outputs
For each scalar target:
- `peak_amplitude`
- `auc_120_abs`
- `iauc_120`

Train quantiles:
- Option A: `q05`, `q50`, `q95`
- Option B: `q10`, `q50`, `q90` (lighter interval, can add wider band later)

Serve:
- point estimate: `q50`
- uncertainty band: `[q_low, q_high]`

## Required Work

### 1. Data and Split Definition
- Freeze training dataset version used for scalar models.
- Define leakage-safe split policy (same unit as current scalar modeling).
- Save split manifests for reproducibility.

### 2. Model Training
- Add quantile training path (pinball loss) per target and quantile.
- Keep feature schema identical to current inference interface.
- Train separate models per target per quantile.
- Export artifacts with clear naming, for example:
  - `.../peak_amplitude/q05/model.pt`
  - `.../peak_amplitude/q50/model.pt`
  - `.../peak_amplitude/q95/model.pt`

### 3. Monotonic Quantile Consistency
- Enforce or correct quantile crossing:
  - ensure `q_low <= q50 <= q_high`.
- Add postprocess safeguard if needed.

### 4. Calibration
- Evaluate raw quantile coverage on held-out data.
- Add conformal adjustment layer (recommended) if under/over-coverage appears.
- Store calibration parameters with model artifacts.

### 5. Backend Inference Integration
- Add quantile model loader in backend inference service.
- New method source tags, e.g.:
  - `quantile_raw_v1`
  - `quantile_conformal_v1`
- Keep current sigma path as fallback until migration is complete.

### 6. API Contract
- Keep existing keys for compatibility during transition.
- Add metadata fields:
  - `ci_metadata.method`
  - `ci_metadata.targets[*].quantiles_used`
  - `ci_metadata.targets[*].coverage_validation` (optional summary)

### 7. Frontend Labeling
- Replace “95% CI” wording with:
  - “Model interval (quantile-based)” or
  - “Calibrated prediction interval” (if conformalized).
- Add tooltip text clarifying interval meaning.

### 8. Validation and Acceptance
- Per target on held-out set:
  - coverage near nominal target (e.g., 90% or 95%).
  - narrower interval than legacy method when possible, without losing coverage.
- Regression checks:
  - no API break
  - no NaN interval outputs
  - stable behavior for nonnegative targets

## Deliverables
- Quantile training script/config updates.
- Saved quantile model artifacts.
- Backend loading + prediction integration.
- API metadata updates.
- Frontend wording update.
- Evaluation report (coverage, interval width, calibration).

## Risks
- Quantile crossing without monotonic handling.
- Miscalibrated intervals if training distribution shifts.
- Increased training/inference artifact complexity.

## Rollout Plan
1. Implement quantile training + offline evaluation only.
2. Integrate backend behind feature flag.
3. Run A/B comparison against current sigma method.
4. Switch default to quantile-calibrated intervals after acceptance.
5. Keep sigma fallback for rollback window.

