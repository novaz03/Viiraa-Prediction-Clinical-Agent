from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
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
from .validation import (
    REQUIRED_DEMOGRAPHIC_CATEGORICAL,
    REQUIRED_DEMOGRAPHIC_NUMERIC,
    validate_and_normalize_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models"
DEFAULT_EXAMPLE = REPO_ROOT / "webapp/frontend/sample_raw_input_single_meal.json"
DEFAULT_LOG_PATH = REPO_ROOT / "webapp/logs/predict_requests.jsonl"
FRONTEND_DIR = REPO_ROOT / "webapp/frontend"
HIDDEN_PREMEAL_FIELDS = [
    "pre_glucose_min_180m",
    "pre_glucose_max_180m",
    "pre_glucose_range_180m",
    "pre_glucose_iqr_180m",
    "pre_glucose_std_180m",
    "pre_glucose_cv_180m",
    "pre_glucose_missing_frac",
    "pre_glucose_valid_count",
    "pre_glucose_longest_gap",
    "glucose_slope_180_60",
    "glucose_slope_60_15",
    "glucose_slope_15_0",
    "glucose_slope_recent_minus_early",
    "baseline_glucose_median_30m",
    "baseline_glucose_mean_30m",
    "premeal_baseline_glucose",
]
RAW_STEP_MINUTES_FIXED = float(os.environ.get("SCALAR_MLP_RAW_STEP_MINUTES", "5.0"))
RAW_BASELINE_WINDOW_MINUTES_FIXED = float(os.environ.get("SCALAR_MLP_RAW_BASELINE_WINDOW_MINUTES", "30.0"))
HIST_BINS = 30
TARGET_METRICS = ("auc_120_abs", "iauc_120", "peak_amplitude")

model_root = Path(os.environ.get("SCALAR_MLP_MODEL_ROOT", str(DEFAULT_MODEL_ROOT)))
example_path = Path(os.environ.get("SCALAR_MLP_EXAMPLE_PATH", str(DEFAULT_EXAMPLE)))
log_path = Path(os.environ.get("SCALAR_MLP_LOG_PATH", str(DEFAULT_LOG_PATH)))
service = ScalarMLPInferenceService(model_root=model_root)

app = FastAPI(title="Viiraa Scalar Prediction API", version="0.2.0")
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
        pass


def _read_recent_logs(limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not log_path.exists():
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


def _iter_all_logs() -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_cgmacros_reference_distributions() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build meal-type stratified reference distributions from CGMacros-derived OOF tables.
    Expected path layout:
      <experiment_root>/best_fold_model/<target>/{oof_predictions.parquet, meal_metrics.parquet}
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    exp_root = model_root.parent
    best_fold_root = exp_root / "best_fold_model"
    for metric in TARGET_METRICS:
        metric_out: Dict[str, np.ndarray] = {}
        oof_path = best_fold_root / metric / "oof_predictions.parquet"
        meal_path = best_fold_root / metric / "meal_metrics.parquet"
        if not (oof_path.exists() and meal_path.exists()):
            out[metric] = metric_out
            continue
        try:
            oof = pd.read_parquet(oof_path)
            meal = pd.read_parquet(meal_path)
            merge_keys = [k for k in ("patient_id", "meal_index", "meal_timestamp", "fold") if k in oof.columns and k in meal.columns]
            if merge_keys:
                merged = oof.merge(meal[[*merge_keys, "meal_type"]], on=merge_keys, how="left")
            else:
                merged = oof.copy()
                merged["meal_type"] = "unknown"
            true_col = f"true__{metric}"
            if true_col not in merged.columns:
                out[metric] = metric_out
                continue
            merged["meal_type"] = merged["meal_type"].astype(str).str.strip().str.lower()
            for mt, grp in merged.groupby("meal_type"):
                vals = pd.to_numeric(grp[true_col], errors="coerce").dropna().astype(float).to_numpy()
                if vals.size:
                    metric_out[mt] = vals
        except Exception:
            metric_out = {}
        out[metric] = metric_out
    return out


CGMACROS_REFERENCE = _load_cgmacros_reference_distributions()


def _build_required_features_or_422(req: RawFeatureBuildRequest) -> Dict[str, Any]:
    try:
        raw_feats = build_required_features_from_raw(
            meal_info=req.meal_info,
            pre_glucose_series=req.pre_glucose_series,
            step_minutes=RAW_STEP_MINUTES_FIXED,
            baseline_window_minutes=RAW_BASELINE_WINDOW_MINUTES_FIXED,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build features: {exc}") from exc
    try:
        return validate_and_normalize_features(raw_feats)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _cohort_values_from_logs(metric: str, meal_type: str) -> np.ndarray:
    out = []
    for row in _iter_all_logs():
        if row.get("status") != "ok":
            continue
        preds = row.get("predictions") or {}
        if metric not in preds:
            continue
        row_meal_type = str((row.get("features") or {}).get("meal_type", "")).lower()
        if row_meal_type and row_meal_type != str(meal_type).lower():
            continue
        try:
            out.append(float(preds[metric]))
        except Exception:
            continue
    return np.asarray(out, dtype=np.float64)


def _histogram_for_value(values: np.ndarray, value: float, n_bins: int = HIST_BINS) -> Dict[str, Any]:
    if values.size == 0:
        values = np.asarray([value], dtype=np.float64)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = min(float(value), 0.0)
        hi = max(float(value) * 1.25 + 1.0, 1.0)
    counts, edges = np.histogram(values, bins=int(n_bins), range=(lo, hi))
    if value <= edges[0]:
        idx = 0
    elif value >= edges[-1]:
        idx = int(n_bins) - 1
    else:
        idx = int(np.searchsorted(edges, value, side="right") - 1)
    idx = int(max(0, min(int(n_bins) - 1, idx)))
    pct = float(100.0 * np.mean(values <= value))
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "bin_counts": [int(x) for x in counts.tolist()],
        "highlight_bin": idx,
        "percentile": pct,
    }


def _build_cohort_comparison(meal_type: str, preds: Dict[str, float]) -> Dict[str, Any]:
    mt = str(meal_type).strip().lower()
    out: Dict[str, Any] = {"meal_type": meal_type, "source": "cgmacros_oof_true_distribution_v1", "metrics": {}}
    for metric, val in preds.items():
        cg_vals = CGMACROS_REFERENCE.get(metric, {}).get(mt, np.asarray([], dtype=np.float64))
        local_vals = _cohort_values_from_logs(metric, meal_type)
        if cg_vals.size > 0:
            cohort_vals = cg_vals
            source = "cgmacros_oof_true"
        elif local_vals.size > 0:
            cohort_vals = local_vals
            source = "local_logs_fallback"
        else:
            cohort_vals = np.asarray([float(val)], dtype=np.float64)
            source = "singleton_fallback"
        hist = _histogram_for_value(cohort_vals, float(val), n_bins=HIST_BINS)
        hist["sample_size"] = int(cohort_vals.size)
        hist["source"] = source
        out["metrics"][metric] = hist
    return out


def _build_personal_comparison(user_id: str, meal_type: str, preds: Dict[str, float]) -> Dict[str, Any]:
    rows = []
    for row in _iter_all_logs():
        if row.get("status") != "ok":
            continue
        if str(row.get("user_id", "anonymous")) != str(user_id):
            continue
        row_meal_type = str((row.get("features") or {}).get("meal_type", "")).lower()
        if row_meal_type != str(meal_type).lower():
            continue
        rp = row.get("predictions") or {}
        if all(k in rp for k in preds.keys()):
            rows.append(rp)
    out: Dict[str, Any] = {"user_id": user_id, "meal_type": meal_type, "history_count": len(rows), "metrics": {}}
    for metric, val in preds.items():
        vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
        if vals.size == 0:
            out["metrics"][metric] = {
                "delta_vs_recent": None,
                "delta_vs_median": None,
                "percentile": None,
                "histogram": None,
            }
            continue
        recent = float(vals[-1])
        median = float(np.median(vals))
        percentile = float(100.0 * np.mean(vals <= float(val)))
        hist = _histogram_for_value(vals, float(val), n_bins=HIST_BINS) if vals.size >= 3 else None
        out["metrics"][metric] = {
            "delta_vs_recent": float(val - recent),
            "delta_vs_median": float(val - median),
            "percentile": percentile,
            "histogram": hist,
        }
    return out


def _risk_band(peak: float, peak_ci95_hi: float) -> str:
    if peak_ci95_hi >= 70 or peak >= 60:
        return "high"
    if peak_ci95_hi >= 50 or peak >= 40:
        return "moderate"
    return "lower"


def _build_analysis_text(
    preds: Dict[str, float],
    intervals: Dict[str, Dict[str, list[float]]],
    cohort: Dict[str, Any],
    personal: Dict[str, Any],
) -> Dict[str, str]:
    peak = float(preds.get("peak_amplitude", 0.0))
    peak_ci95 = intervals.get("peak_amplitude", {}).get("ci_95", [peak, peak])
    band = _risk_band(peak=peak, peak_ci95_hi=float(peak_ci95[1]))

    if band == "high":
        short_term = "Predicted post-meal excursion is elevated and may reflect a significant glucose spike in the next 2 hours."
        headline = "High predicted excursion profile."
    elif band == "moderate":
        short_term = "Predicted post-meal excursion is moderate, with noticeable glucose rise expected."
        headline = "Moderate predicted excursion profile."
    else:
        short_term = "Predicted post-meal excursion is relatively lower for this input profile."
        headline = "Lower predicted excursion profile."

    c_auc = float(cohort.get("metrics", {}).get("auc_120_abs", {}).get("percentile", 50.0))
    c_iauc = float(cohort.get("metrics", {}).get("iauc_120", {}).get("percentile", 50.0))
    c_peak = float(cohort.get("metrics", {}).get("peak_amplitude", {}).get("percentile", 50.0))
    c_avg = (c_auc + c_iauc + c_peak) / 3.0
    if c_avg >= 75:
        cohort_text = "Compared with typical meals of the same type, this prediction is in a higher excursion range."
    elif c_avg <= 35:
        cohort_text = "Compared with typical meals of the same type, this prediction is in a lower excursion range."
    else:
        cohort_text = "Compared with typical meals of the same type, this prediction is near the middle range."

    p_metrics = personal.get("metrics", {})
    deltas = []
    for metric in ("auc_120_abs", "iauc_120", "peak_amplitude"):
        d = p_metrics.get(metric, {}).get("delta_vs_median")
        if d is not None:
            deltas.append(float(d))
    if not deltas:
        personal_text = "No prior same-meal history is available yet for personal comparison."
    elif np.mean(deltas) > 0:
        personal_text = "Compared with your recent meals of the same type, this appears somewhat higher."
    else:
        personal_text = "Compared with your recent meals of the same type, this appears somewhat lower."

    return {
        "headline": headline,
        "short_term_impact": short_term,
        "cohort_comparison": cohort_text,
        "personal_comparison": personal_text,
        "safety_note": "Research-use prediction only. This is not diagnosis or treatment advice.",
    }


def _cgmacros_reference_meal_types() -> list[str]:
    mts = set()
    for metric_map in CGMACROS_REFERENCE.values():
        mts.update(metric_map.keys())
    return sorted(mts)


def _build_hist_artifact(metric: str, meal_type: str) -> Dict[str, Any]:
    mt = str(meal_type).strip().lower()
    vals = CGMACROS_REFERENCE.get(metric, {}).get(mt, np.asarray([], dtype=np.float64))
    if vals.size == 0:
        return {
            "metric": metric,
            "meal_type": meal_type,
            "available": False,
            "sample_size": 0,
            "histogram": None,
        }
    hist = _histogram_for_value(vals, float(np.median(vals)), n_bins=HIST_BINS)
    return {
        "metric": metric,
        "meal_type": meal_type,
        "available": True,
        "sample_size": int(vals.size),
        "histogram": hist,
    }


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
        payload = service.load_example(example_path)
        return {"raw_input": payload}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load example input: {exc}") from exc


@app.get("/v1/example-response")
def example_response() -> Dict[str, Any]:
    try:
        raw_payload = service.load_example(example_path)
        features = _build_required_features_or_422(RawFeatureBuildRequest(**raw_payload))
        preds, intervals, _ = service.predict_with_intervals(features)
        meal_type = str(features.get("meal_type", "Lunch"))
        cohort = _build_cohort_comparison(meal_type=meal_type, preds=preds)
        personal = _build_personal_comparison(user_id="example-user", meal_type=meal_type, preds=preds)
        analysis_text = _build_analysis_text(preds=preds, intervals=intervals, cohort=cohort, personal=personal)
        return {
            "predictions": preds,
            "prediction_intervals": intervals,
            "cohort_comparison": cohort,
            "personal_comparison": personal,
            "analysis_text": analysis_text,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build example response: {exc}") from exc


@app.get("/v1/reference-histograms")
def reference_histograms(meal_type: str, include_values: bool = False, max_values: int = 200) -> Dict[str, Any]:
    mt = str(meal_type).strip().lower()
    if not mt:
        raise HTTPException(status_code=422, detail="meal_type is required.")
    out: Dict[str, Any] = {"meal_type": mt, "source": "cgmacros_oof_true_distribution_v1", "bins": HIST_BINS, "metrics": {}}
    for metric in TARGET_METRICS:
        artifact = _build_hist_artifact(metric, mt)
        if include_values and artifact["available"]:
            vals = CGMACROS_REFERENCE.get(metric, {}).get(mt, np.asarray([], dtype=np.float64))
            n = max(0, min(int(max_values), int(vals.size)))
            artifact["sample_values"] = [float(x) for x in vals[:n].tolist()]
        out["metrics"][metric] = artifact
    return out


@app.get("/v1/reference-histograms/meal-types")
def reference_histogram_meal_types() -> Dict[str, Any]:
    return {"meal_types": _cgmacros_reference_meal_types(), "source": "cgmacros_oof_true_distribution_v1"}


@app.get("/v1/expected-columns")
def expected_columns() -> Dict[str, Any]:
    return service.expected_columns()


@app.get("/v1/ui-config")
def ui_config() -> Dict[str, Any]:
    return {
        "fixed_raw_params": {
            "step_minutes": RAW_STEP_MINUTES_FIXED,
            "baseline_window_minutes": RAW_BASELINE_WINDOW_MINUTES_FIXED,
        },
        "hidden_premeal_fields": HIDDEN_PREMEAL_FIELDS,
        "demographic_fields": service.demographic_fields(),
        "required_demographic_fields": {
            "numeric": REQUIRED_DEMOGRAPHIC_NUMERIC,
            "categorical": REQUIRED_DEMOGRAPHIC_CATEGORICAL,
        },
    }


@app.post("/v1/build-features-from-raw", response_model=RawFeatureBuildResponse)
def build_features_from_raw(req: RawFeatureBuildRequest) -> RawFeatureBuildResponse:
    return RawFeatureBuildResponse(features=_build_required_features_or_422(req))


@app.post("/v1/diagnostics/raw-input")
def diagnostics_raw_input(req: RawFeatureBuildRequest) -> Dict[str, Any]:
    features = _build_required_features_or_422(req)
    preds, intervals, ci_method = service.predict_with_intervals(features)
    return {
        "raw_contract": {
            "strict_required_fields": True,
            "default_fill_for_required_raw_fields": False,
        },
        "features_used_for_model": features,
        "predictions": preds,
        "prediction_intervals": intervals,
        "ci_method": ci_method,
    }


@app.get("/v1/logs/recent")
def recent_logs(limit: int = 20) -> Dict[str, Any]:
    n = max(1, min(int(limit), 200))
    return {"items": _read_recent_logs(n)}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    features = _build_required_features_or_422(req.raw_input)

    try:
        preds, intervals, ci_method = service.predict_with_intervals(features)
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

    meal_type = str(features.get("meal_type", "Lunch"))
    cohort = _build_cohort_comparison(meal_type=meal_type, preds=preds)
    personal = _build_personal_comparison(user_id=req.user_id, meal_type=meal_type, preds=preds)
    analysis_text = _build_analysis_text(preds=preds, intervals=intervals, cohort=cohort, personal=personal)

    notes = [
        "This endpoint predicts scalar targets from provided meal/glucose context.",
        "Accepts a single user-facing raw payload (`meal_info` + `pre_glucose_series`) and computes engineered features server-side.",
        "Confidence intervals use fold-ensemble quantile intervals when fold checkpoints are available.",
        "Cohort comparison uses CGMacros-derived same-meal-type OOF distributions with 30-bin histograms.",
        "Personal comparison includes user-history percentiles and 30-bin histograms when enough history is available.",
        "AUC-abs and peak-amplitude predictions are constrained to nonnegative values in postprocessing.",
        "Use for research workflows; not for diagnosis or treatment decisions.",
    ]

    expected = service.expected_columns() if req.include_expected_columns else None
    resp = PredictResponse(
        request_id=request_id,
        predictions=preds,
        prediction_intervals=intervals,
        cohort_comparison=cohort,
        personal_comparison=personal,
        analysis_text=analysis_text,
        model_family="scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models",
        notes=notes,
        features_used=features,
        expected_columns=expected,
    )
    _append_log(
        {
            "ts_utc": _utc_now(),
            "request_id": request_id,
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "predictions": preds,
            "prediction_intervals": intervals,
            "features": {"meal_type": meal_type},
            "user_id": req.user_id,
            "ci_method": ci_method,
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
            features = _build_required_features_or_422(item)
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
        "Batch endpoint predicts scalar targets for each raw input object in `items`.",
        "Each item requires `meal_info` + `pre_glucose_series`; engineered features are computed server-side.",
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
