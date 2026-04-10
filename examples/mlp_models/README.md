# MLP Example Inputs

These example payloads are for the scalar MLP checkpoints in:

- `outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models/`

Files:

- `sample_input_single_meal.csv`
- `sample_input_single_meal.json`
- `sample_raw_input_single_meal.json`

Notes:

- This is a single-meal example row intended for inference smoke tests.
- Column names follow the exported MLP checkpoint feature schema (meal context + pre-meal glucose/CWT-derived engineered features).
- Engineered sample values are aligned to current serving conventions:
  - `Height` in inches
  - `Body weight` in pounds
  - `pct_macro_cals_*` as fractions in `[0, 1]`
  - `pre_glucose_valid_count` in minute-scale coverage (e.g., `36 * 5 = 180`)
- If your inference code computes derived columns (log, pct macro, interactions) internally, you can ignore those fields in this sample and provide only raw inputs required by your pipeline.
- The raw sample is compatible with `POST /v1/build-features-from-raw` and `POST /v1/predict` (raw-input contract).
