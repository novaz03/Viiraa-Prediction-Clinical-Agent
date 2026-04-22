# Viiraa Prediction Models, API, Samples, and Hugging Face Upload

This README summarizes:
- which models are used for prediction,
- where sample inputs/outputs live,
- how to call the prediction API,
- and how to upload the used model artifacts to Hugging Face.

## 1) Models used for prediction

The web API serves scalar MLP checkpoints for three targets:
- `auc_120_abs`
- `iauc_120`
- `peak_amplitude`

Model classes supported by inference:
- `ScalarMLP`
- `ResidualScalarMLP`
- `GatedScalarMLP`

Relevant code:
- `webapp/backend/inference.py` (`TARGETS`, model class imports, checkpoint loading)
- `webapp/backend/app.py` (model root discovery/registry and model selection)

### Model sets discovered by the API

The backend auto-discovers model roots by ID:
- `default` (from `SCALAR_MLP_MODEL_ROOT` if set; otherwise default path below)
- `original`
- `cwt_target`

Built-in sibling experiment paths:
- `../Viiraa-Prediction/outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_original/final_models`
- `../Viiraa-Prediction/outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models`

The deployment scripts currently set `SCALAR_MLP_MODEL_ROOT` to the `to_original/final_models` path, so that is the default used model set unless overridden.

## 2) Sample input and sample output

### Sample input files

In this repo:
- `examples/mlp_models/sample_raw_input_single_meal.json`
- `examples/mlp_models/sample_input_single_meal.json`
- `examples/mlp_models/sample_input_single_meal.csv`

Also mirrored for frontend loading:
- `webapp/frontend/sample_raw_input_single_meal.json`

### Sample output

You can fetch an end-to-end sample prediction response from:
- `GET /v1/example-response`

This endpoint builds features from the sample raw input, runs prediction, and returns:
- `predictions`
- `prediction_intervals`
- `ci_metadata`
- `cohort_comparison`
- `personal_comparison`
- `analysis_text`

## 3) API for calling the models

Primary endpoints:
- `POST /v1/predict` (single raw-input request)
- `POST /v1/predict/batch` (multiple raw-input items)

Useful companion endpoints:
- `GET /v1/healthz` (active model info + available model IDs)
- `GET /v1/example-input`
- `GET /v1/example-response`
- `GET /v1/ui-config` (includes `default_model_id` and `model_options`)
- `GET /v1/expected-columns`
- `POST /v1/build-features-from-raw`
- `POST /v1/diagnostics/raw-input`

Request/response examples are documented in:
- `webapp/README.md`

## 4) Uploading used models to Hugging Face

Upload script:
- `scripts/publish/upload_scalar_mlp_models_to_hf.py`

What it does:
- creates/updates an HF model repo,
- uploads all files under the selected model folder,
- writes/updates a basic model card (`README.md`) in HF.

### Prerequisites

1. Install dependency:
```bash
pip install huggingface_hub
```

2. Set HF token:
```bash
export HF_TOKEN=your_write_token_here
```

### Upload commands

Upload the currently used default (`to_original`) model set:
```bash
python3 scripts/publish/upload_scalar_mlp_models_to_hf.py \
  --repo-id <your-hf-username-or-org>/<repo-name> \
  --source-dir ../Viiraa-Prediction/outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_original/final_models
```

Upload the `cwt_target` model set:
```bash
python3 scripts/publish/upload_scalar_mlp_models_to_hf.py \
  --repo-id <your-hf-username-or-org>/<repo-name>-cwt-target \
  --source-dir ../Viiraa-Prediction/outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models
```

Optional flags:
- `--private` to keep repo private
- `--token-env <ENV_VAR_NAME>` if token is in a different variable
- `--commit-message "..."` to set custom upload commit message

### Recommended artifact layout in HF

Keep one HF repo per model set:
- `.../viiraa-scalar-mlp-original`
- `.../viiraa-scalar-mlp-cwt-target`

Each repo will contain per-target folders such as:
- `auc_120_abs/model.pt`
- `iauc_120/model.pt`
- `peak_amplitude/model.pt`

## 5) Quick checks after deployment

Use:
```bash
curl -s http://localhost:8000/v1/healthz
```

Confirm:
- `active_model_id`
- `model_root`
- `available_models`

Then call:
```bash
curl -s http://localhost:8000/v1/example-response
```

to verify inference and response contract.
