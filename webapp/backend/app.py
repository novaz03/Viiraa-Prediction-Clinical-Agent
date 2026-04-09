from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .feature_engineering import build_required_features_from_raw
from .inference import ScalarMLPInferenceService
from .schemas import (
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    RawFeatureBuildRequest,
    RawFeatureBuildResponse,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models"
DEFAULT_EXAMPLE = REPO_ROOT / "examples/mlp_models/sample_input_single_meal.json"
DEFAULT_LOG_PATH = REPO_ROOT / "webapp/logs/predict_requests.jsonl"
FRONTEND_DIR = REPO_ROOT / "webapp/frontend"
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

model_root = Path(os.environ.get("SCALAR_MLP_MODEL_ROOT", str(DEFAULT_MODEL_ROOT)))
example_path = Path(os.environ.get("SCALAR_MLP_EXAMPLE_PATH", str(DEFAULT_EXAMPLE)))
log_path = Path(os.environ.get("SCALAR_MLP_LOG_PATH", str(DEFAULT_LOG_PATH)))
service = ScalarMLPInferenceService(model_root=model_root)

app = FastAPI(title="Viiraa Scalar Prediction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(payload: Dict[str, Any]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        # Do not fail prediction path because of logging issues.
        pass


def _read_recent_logs(limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if not log_path.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=int(limit))
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return list(rows)[::-1]


def _validate_and_normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="`features` must be a JSON object.")

    missing = [k for k in REQUIRED_FIELDS if k not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {missing}")

    out = dict(features)
    for col in NUMERIC_REQUIRED:
        try:
            out[col] = float(out[col])
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Field `{col}` must be numeric.") from exc
    out["meal_type"] = str(out["meal_type"])
    return out


def _build_required_features_or_422(req: RawFeatureBuildRequest) -> Dict[str, Any]:
    try:
        raw_feats = build_required_features_from_raw(
            meal_info=req.meal_info,
            pre_glucose_series=req.pre_glucose_series,
            step_minutes=float(req.step_minutes),
            baseline_window_minutes=float(req.baseline_window_minutes),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build features: {exc}") from exc
    return _validate_and_normalize_features(raw_feats)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = FRONTEND_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/v1/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/example-input")
def get_example_input() -> Dict[str, Any]:
    try:
        return {"features": service.load_example(example_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load example input: {exc}") from exc


@app.get("/v1/expected-columns")
def expected_columns() -> Dict[str, Any]:
    return service.expected_columns()


@app.post("/v1/build-features-from-raw", response_model=RawFeatureBuildResponse)
def build_features_from_raw(req: RawFeatureBuildRequest) -> RawFeatureBuildResponse:
    return RawFeatureBuildResponse(features=_build_required_features_or_422(req))


@app.get("/v1/logs/recent")
def recent_logs(limit: int = 20) -> Dict[str, Any]:
    n = max(1, min(int(limit), 200))
    return {"items": _read_recent_logs(n)}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    if req.raw_input is not None:
        features = _build_required_features_or_422(req.raw_input)
    else:
        features = _validate_and_normalize_features(req.features)

    try:
        preds = service.predict_one(features)
    except Exception as exc:
        _append_log(
            {
                "ts_utc": _utc_now(),
                "request_id": request_id,
                "status": "error",
                "error": str(exc),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        )
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    notes = [
        "This endpoint predicts scalar targets from provided meal/glucose context.",
        "Supports either `features` (engineered fields) or `raw_input` (meal info + pre-glucose sequence).",
        "Additional non-required fields are imputed/derived using model preprocessing defaults.",
        "Use for research workflows; not for diagnosis or treatment decisions.",
    ]

    expected = service.expected_columns() if req.include_expected_columns else None
    resp = PredictResponse(
        request_id=request_id,
        predictions=preds,
        model_family="scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models",
        notes=notes,
        features_used=features if req.raw_input is not None else None,
        expected_columns=expected,
    )
    _append_log(
        {
            "ts_utc": _utc_now(),
            "request_id": request_id,
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "predictions": preds,
        }
    )
    return resp


@app.post("/v1/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    if not req.items:
        raise HTTPException(status_code=422, detail="`items` must contain at least one feature object.")

    preds_all = []
    try:
        for item in req.items:
            features = _validate_and_normalize_features(item)
            preds_all.append(service.predict_one(features))
    except HTTPException:
        raise
    except Exception as exc:
        _append_log(
            {
                "ts_utc": _utc_now(),
                "request_id": request_id,
                "status": "error",
                "error": str(exc),
                "batch_size": len(req.items),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        )
        raise HTTPException(status_code=400, detail=f"Batch inference failed: {exc}") from exc

    notes = [
        "Batch endpoint predicts scalar targets for each feature object in `items`.",
        "All items must satisfy required input fields and numeric typing.",
        "Use for research workflows; not for diagnosis or treatment decisions.",
    ]
    expected = service.expected_columns() if req.include_expected_columns else None
    resp = PredictBatchResponse(
        request_id=request_id,
        model_family="scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models",
        notes=notes,
        predictions=preds_all,
        expected_columns=expected,
    )
    _append_log(
        {
            "ts_utc": _utc_now(),
            "request_id": request_id,
            "status": "ok",
            "batch_size": len(req.items),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
    )
    return resp
