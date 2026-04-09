# Viiraa Scalar Prediction Web App (MVP)

Phase 1 scaffold for a website that accepts meal/glucose context and predicts:
- `auc_120_abs`
- `iauc_120`
- `peak_amplitude`

## Structure
- `backend/app.py`: FastAPI app + endpoints + static frontend serving.
- `backend/inference.py`: loads scalar MLP checkpoints and runs inference.
- `frontend/index.html`: simple UI for JSON input, example loading, and predictions.

## API Endpoints
- `GET /v1/healthz`
- `GET /v1/example-input`
- `GET /v1/expected-columns`
- `POST /v1/build-features-from-raw`
- `POST /v1/predict`
- `POST /v1/predict/batch`

`POST /v1/predict` body (engineered features mode):
```json
{
  "features": {
    "meal_type": "Lunch",
    "meal_calories": 650,
    "...": 0
  },
  "include_expected_columns": false
}
```

`POST /v1/build-features-from-raw` body:
```json
{
  "meal_info": {
    "meal_type": "Lunch",
    "meal_calories": 650,
    "carbs_g": 70,
    "protein_g": 32,
    "fat_g": 22,
    "A1c PDL (Lab)": 5.7,
    "Fasting GLU - PDL (Lab)": 95,
    "minutes_since_last_meal": 180
  },
  "pre_glucose_series": [94, 93, 92, 92, 93, 94],
  "step_minutes": 5,
  "baseline_window_minutes": 30
}
```

`POST /v1/predict` body (raw mode, one-call derive + predict):
```json
{
  "raw_input": {
    "meal_info": {
      "meal_type": "Lunch",
      "meal_calories": 650,
      "carbs_g": 70,
      "protein_g": 32,
      "fat_g": 22,
      "A1c PDL (Lab)": 5.7,
      "Fasting GLU - PDL (Lab)": 95,
      "minutes_since_last_meal": 180
    },
    "pre_glucose_series": [94, 93, 92, 92, 93, 94],
    "step_minutes": 5,
    "baseline_window_minutes": 30
  },
  "include_expected_columns": false
}
```

`POST /v1/predict/batch` body:
```json
{
  "items": [
    { "meal_type": "Lunch", "meal_calories": 650, "...": 0 },
    { "meal_type": "Dinner", "meal_calories": 540, "...": 0 }
  ],
  "include_expected_columns": false
}
```

## Required Fields (current strict contract)
- `meal_type`
- `meal_calories`
- `carbs_g`
- `protein_g`
- `fat_g`
- `A1c PDL (Lab)`
- `Fasting GLU - PDL (Lab)`
- `minutes_since_last_meal`
- `baseline_glucose_median_30m`
- `baseline_glucose_mean_30m`
- `pre_glucose_min_180m`
- `pre_glucose_max_180m`
- `pre_glucose_range_180m`
- `pre_glucose_iqr_180m`
- `pre_glucose_std_180m`
- `pre_glucose_cv_180m`
- `glucose_slope_180_60`
- `glucose_slope_60_15`
- `glucose_slope_15_0`
- `glucose_slope_recent_minus_early`
- `pre_glucose_missing_frac`
- `pre_glucose_valid_count`
- `pre_glucose_longest_gap`
- `premeal_baseline_glucose`

## Run (when environment has required deps)
```bash
pip install -r webapp/requirements.txt
uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://localhost:8000/`

## Run on Compute Node (LSF + port mapping)

Use the submit helper (requires explicit port argument):

```bash
bash scripts/bsub/web/submit_viiraa_webapp.sh --port 8000
```

This wrapper exports required variables (startup-style):
- `SHADOW`
- `LSF_DOCKER_VOLUMES`
- `LSF_DOCKER_SHM_SIZE`
- `LSF_DOCKER_PORTS=<port>:<port>`
- `VIIRAA_WEBAPP_PORT=<port>`

The job log (`logs/viiraa_webapp_api.<JOBID>.out`) prints the SSH tunnel command template.

## Notes
- Current MVP enforces required core fields and imputes/derives additional non-required fields.
- Derived fields (log and selected interactions) are auto-generated when possible.
- Raw mode currently derives core glucose summary/slope features from pre-meal glucose sequence.
- For research use only; not for diagnosis/treatment decisions.
- Requests are logged to `webapp/logs/predict_requests.jsonl` (configurable via `SCALAR_MLP_LOG_PATH`).
