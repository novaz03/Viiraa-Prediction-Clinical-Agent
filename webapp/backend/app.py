from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
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
PRED_REPO_ROOT = REPO_ROOT.parent / "Viiraa-Prediction"


def _load_secret_env() -> None:
    """
    Lightweight .secret loader (dotenv-style, no extra dependency).
    Existing environment variables always take precedence over file values.
    """
    candidates = [
        REPO_ROOT / ".secret",
        REPO_ROOT / "webapp/.secret",
    ]
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip("'").strip('"')
                if not key:
                    continue
                os.environ.setdefault(key, val)
        except Exception:
            continue


_load_secret_env()

DEFAULT_MODEL_ROOT = (
    PRED_REPO_ROOT / "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_original/final_models"
)
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
ANALYSIS_PROVIDER = str(os.environ.get("VIIRAA_ANALYSIS_PROVIDER", "heuristic")).strip().lower()
ANALYSIS_MODEL = str(os.environ.get("VIIRAA_ANALYSIS_MODEL", "gpt-4.1-mini")).strip()
ANALYSIS_MAX_OUTPUT_TOKENS = int(os.environ.get("VIIRAA_ANALYSIS_MAX_OUTPUT_TOKENS", "450"))
ANALYSIS_API_BASE = str(os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
GEMINI_API_BASE = str(
    os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
).rstrip("/")
LOCAL_ANALYSIS_URL = str(os.environ.get("VIIRAA_LOCAL_ANALYSIS_URL", "http://127.0.0.1:11434/api/generate")).strip()
ANALYSIS_HTTP_TIMEOUT_SECONDS = float(os.environ.get("VIIRAA_ANALYSIS_HTTP_TIMEOUT_SECONDS", "20"))


def _infer_model_family_from_root(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(REPO_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return str(p)


def _safe_model_id(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s).strip().lower())
    return out or "model"


def _model_targets_exist(root: Path) -> bool:
    try:
        for t in TARGET_METRICS:
            if not (root / t / "model.pt").exists():
                return False
        return True
    except Exception:
        return False


def _discover_model_roots() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    explicit_root = Path(os.environ.get("SCALAR_MLP_MODEL_ROOT", str(DEFAULT_MODEL_ROOT)))
    out["default"] = explicit_root

    # Built-in sibling experiments.
    candidate = {
        "original": PRED_REPO_ROOT / "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_original/final_models",
        "cwt_target": PRED_REPO_ROOT / "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models",
    }
    for k, p in candidate.items():
        out.setdefault(k, p)

    # Optional override list: "id=/abs/path;id2=/abs/path2"
    raw = os.environ.get("SCALAR_MLP_MODEL_OPTIONS", "").strip()
    if raw:
        for part in [x for x in raw.split(";") if x.strip()]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            key = _safe_model_id(k)
            val = Path(v.strip())
            if v.strip():
                out[key] = val
    return out


def _build_model_registry() -> tuple[Dict[str, Dict[str, Any]], str]:
    discovered = _discover_model_roots()
    reg: Dict[str, Dict[str, Any]] = {}
    seen_paths: Dict[Path, str] = {}
    preferred_order = ["default", "original", "cwt_target"]
    ordered_keys = [k for k in preferred_order if k in discovered] + [k for k in discovered.keys() if k not in preferred_order]
    for model_id in ordered_keys:
        root = discovered[model_id]
        if not _model_targets_exist(root):
            continue
        try:
            root_key = root.resolve()
        except Exception:
            root_key = root
        if root_key in seen_paths:
            continue
        family = _infer_model_family_from_root(root)
        try:
            reg[model_id] = {
                "model_id": model_id,
                "model_root": root,
                "model_family": family,
                "service": ScalarMLPInferenceService(model_root=root),
            }
            seen_paths[root_key] = model_id
        except Exception:
            continue
    if not reg:
        raise RuntimeError("No valid scalar model roots found (missing <target>/model.pt checkpoints).")

    preferred = _safe_model_id(os.environ.get("SCALAR_MLP_MODEL_ID", "default"))
    active = preferred if preferred in reg else ("default" if "default" in reg else next(iter(reg.keys())))
    return reg, active


def _model_display_name(model_id: str, model_family: str) -> str:
    key = _safe_model_id(model_id)
    if key == "original":
        return "Original target"
    if key == "cwt_target":
        return "CWT-smoothed target"
    if key == "default":
        fam = str(model_family).lower()
        if "to_original" in fam:
            return "Original target (default)"
        if "to_cwttarget" in fam or "to_cwt_target" in fam:
            return "CWT-smoothed target (default)"
        return "Default model"
    return key.replace("_", " ").strip().title() or "Model"


MODEL_REGISTRY, ACTIVE_MODEL_ID = _build_model_registry()
_active = MODEL_REGISTRY[ACTIVE_MODEL_ID]
model_root = Path(_active["model_root"])
model_family = str(_active["model_family"])
service = _active["service"]
example_path = Path(os.environ.get("SCALAR_MLP_EXAMPLE_PATH", str(DEFAULT_EXAMPLE)))
log_path = Path(os.environ.get("SCALAR_MLP_LOG_PATH", str(DEFAULT_LOG_PATH)))


def _resolve_model_bundle(model_id: str | None) -> Dict[str, Any]:
    if model_id is None or not str(model_id).strip():
        return MODEL_REGISTRY[ACTIVE_MODEL_ID]
    key = _safe_model_id(str(model_id))
    bundle = MODEL_REGISTRY.get(key)
    if bundle is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model_id '{model_id}'. Available: {sorted(MODEL_REGISTRY.keys())}",
        )
    return bundle

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
        if cg_vals.size > 0:
            cohort_vals = cg_vals
            source = "cgmacros_oof_true"
        else:
            out["metrics"][metric] = {
                "available": False,
                "sample_size": 0,
                "source": "cgmacros_oof_true",
                "reason": "No CGMacros reference rows for this meal_type/metric.",
            }
            continue
        hist = _histogram_for_value(cohort_vals, float(val), n_bins=HIST_BINS)
        hist["available"] = True
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
    features: Dict[str, Any],
    preds: Dict[str, float],
    intervals: Dict[str, Dict[str, list[float]]],
    cohort: Dict[str, Any],
    personal: Dict[str, Any],
    ci_metadata: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    ai = _build_analysis_text_with_ai(
        features=features,
        preds=preds,
        intervals=intervals,
        cohort=cohort,
        personal=personal,
        ci_metadata=ci_metadata or {},
    )
    if ai is not None:
        return ai
    return _build_analysis_text_heuristic(preds=preds, intervals=intervals, cohort=cohort, personal=personal)


def _build_analysis_text_heuristic(
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
        "analysis_source": "heuristic",
    }


def _build_anonymized_analysis_payload(
    features: Dict[str, Any],
    preds: Dict[str, float],
    intervals: Dict[str, Dict[str, list[float]]],
    cohort: Dict[str, Any],
    personal: Dict[str, Any],
    ci_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    demo = {
        "Age": features.get("Age"),
        "Gender": features.get("Gender"),
        "BMI": features.get("BMI"),
        "Height_in": features.get("Height"),
        "Body_weight_lb": features.get("Body weight"),
        "A1c": features.get("A1c PDL (Lab)"),
        "Fasting_GLU": features.get("Fasting GLU - PDL (Lab)"),
    }
    return {
        "meal_type": features.get("meal_type"),
        "demographics_labs": demo,
        "predictions": {
            "peak_glucose_abs": preds.get("peak_glucose_abs"),
            "peak_amplitude": preds.get("peak_amplitude"),
            "auc_120_abs": preds.get("auc_120_abs"),
            "iauc_120": preds.get("iauc_120"),
        },
        "intervals": {
            "peak_glucose_abs_ci95": (intervals.get("peak_glucose_abs") or {}).get("ci_95"),
            "peak_amplitude_ci95": (intervals.get("peak_amplitude") or {}).get("ci_95"),
            "auc_120_abs_ci95": (intervals.get("auc_120_abs") or {}).get("ci_95"),
            "iauc_120_ci95": (intervals.get("iauc_120") or {}).get("ci_95"),
        },
        "cohort_quantiles": {
            "peak_percentile": ((cohort.get("metrics") or {}).get("peak_amplitude") or {}).get("percentile"),
            "auc_percentile": ((cohort.get("metrics") or {}).get("auc_120_abs") or {}).get("percentile"),
            "iauc_percentile": ((cohort.get("metrics") or {}).get("iauc_120") or {}).get("percentile"),
        },
        "personal_comparison": {
            "history_count": personal.get("history_count"),
            "peak_delta_vs_median": ((personal.get("metrics") or {}).get("peak_amplitude") or {}).get("delta_vs_median"),
            "auc_delta_vs_median": ((personal.get("metrics") or {}).get("auc_120_abs") or {}).get("delta_vs_median"),
            "iauc_delta_vs_median": ((personal.get("metrics") or {}).get("iauc_120") or {}).get("delta_vs_median"),
        },
        "ci_metadata": ci_metadata,
    }


def _call_openai_analysis(payload: Dict[str, Any]) -> Dict[str, str] | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    system_prompt, user_prompt = _analysis_prompts(payload)

    req_body = {
        "model": ANALYSIS_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(user_prompt)}]},
        ],
        "max_output_tokens": ANALYSIS_MAX_OUTPUT_TOKENS,
    }
    req = urllib.request.Request(
        f"{ANALYSIS_API_BASE}/responses",
        data=json.dumps(req_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=ANALYSIS_HTTP_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    text = None
    if isinstance(obj, dict):
        if isinstance(obj.get("output_text"), str):
            text = obj.get("output_text")
        elif isinstance(obj.get("output"), list):
            # Fallback parser for structured output arrays.
            chunks = []
            for item in obj.get("output", []):
                for c in item.get("content", []) if isinstance(item, dict) else []:
                    t = c.get("text") if isinstance(c, dict) else None
                    if isinstance(t, str):
                        chunks.append(t)
            if chunks:
                text = "\n".join(chunks)
    return _finalize_analysis_output(text=text, source="ai_openai")


def _analysis_prompts(payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    system_prompt = (
        "You are a clinical research assistant for post-meal glucose analysis. "
        "Use only provided numeric context. Be cautious, concise, and non-diagnostic. "
        "Return strict JSON with keys: headline, short_term_impact, cohort_comparison, personal_comparison, safety_note."
    )
    user_prompt = {
        "task": (
            "Generate interpretation text by combining absolute predicted values, uncertainty intervals, "
            "and relative CGMacros/personal quantile context."
        ),
        "constraints": [
            "No diagnosis or treatment claims.",
            "Reference uncertainty as directional confidence when intervals are wide.",
            "Mention both absolute burden and relative percentile context.",
        ],
        "input": payload,
    }
    return system_prompt, user_prompt


def _strip_json_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return s


def _extract_json_dict(text: str) -> Dict[str, Any] | None:
    s = _strip_json_code_fence(text)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _finalize_analysis_output(text: str | None, source: str) -> Dict[str, str] | None:
    if not text:
        return None
    out = _extract_json_dict(text)
    if out is None:
        return None
    needed = ("headline", "short_term_impact", "cohort_comparison", "personal_comparison", "safety_note")
    if not all(isinstance(out.get(k), str) and out.get(k).strip() for k in needed):
        return None
    out["analysis_source"] = source
    return {k: str(v) for k, v in out.items()}


def _call_gemini_analysis(payload: Dict[str, Any]) -> Dict[str, str] | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    system_prompt, user_prompt = _analysis_prompts(payload)
    model = ANALYSIS_MODEL or "gemini-1.5-flash"
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": json.dumps(user_prompt)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": ANALYSIS_MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
        },
    }
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=ANALYSIS_HTTP_TIMEOUT_SECONDS) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    text = None
    if isinstance(obj, dict):
        cands = obj.get("candidates")
        if isinstance(cands, list) and cands:
            parts = (((cands[0] or {}).get("content") or {}).get("parts")) if isinstance(cands[0], dict) else None
            if isinstance(parts, list):
                chunks = []
                for p in parts:
                    t = p.get("text") if isinstance(p, dict) else None
                    if isinstance(t, str):
                        chunks.append(t)
                if chunks:
                    text = "\n".join(chunks)
    return _finalize_analysis_output(text=text, source="ai_gemini")


def _call_local_analysis(payload: Dict[str, Any]) -> Dict[str, str] | None:
    if not LOCAL_ANALYSIS_URL:
        return None
    system_prompt, user_prompt = _analysis_prompts(payload)
    req_body = {
        "model": ANALYSIS_MODEL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "format": "json",
    }
    req = urllib.request.Request(
        LOCAL_ANALYSIS_URL,
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=ANALYSIS_HTTP_TIMEOUT_SECONDS) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    text = None
    if isinstance(obj, dict):
        for k in ("output_text", "text", "response", "result"):
            if isinstance(obj.get(k), str):
                text = obj.get(k)
                break
    return _finalize_analysis_output(text=text, source="ai_local")


def _build_analysis_text_with_ai(
    features: Dict[str, Any],
    preds: Dict[str, float],
    intervals: Dict[str, Dict[str, list[float]]],
    cohort: Dict[str, Any],
    personal: Dict[str, Any],
    ci_metadata: Dict[str, Any],
) -> Dict[str, str] | None:
    provider = ANALYSIS_PROVIDER
    payload = _build_anonymized_analysis_payload(
        features=features,
        preds=preds,
        intervals=intervals,
        cohort=cohort,
        personal=personal,
        ci_metadata=ci_metadata,
    )
    if provider in {"openai", "ai", "llm"}:
        return _call_openai_analysis(payload)
    if provider == "gemini":
        return _call_gemini_analysis(payload)
    if provider in {"local", "local_model"}:
        return _call_local_analysis(payload)
    return None


def _augment_predictions_for_output(
    preds: Dict[str, float], intervals: Dict[str, Dict[str, list[float]]], features: Dict[str, Any]
) -> tuple[Dict[str, float], Dict[str, Dict[str, list[float]]]]:
    out_preds = dict(preds)
    out_intervals: Dict[str, Dict[str, list[float]]] = {k: dict(v) for k, v in intervals.items()}
    try:
        baseline = float(features.get("premeal_baseline_glucose"))
    except Exception:
        baseline = float("nan")
    if not np.isfinite(baseline):
        return out_preds, out_intervals
    peak_delta = float(preds.get("peak_amplitude", 0.0))
    out_preds["peak_glucose_abs"] = float(baseline + peak_delta)
    peak_ci = intervals.get("peak_amplitude", {}).get("ci_95")
    peak_ci80 = intervals.get("peak_amplitude", {}).get("ci_80")
    if peak_ci80 is not None and len(peak_ci80) == 2:
        out_intervals.setdefault("peak_glucose_abs", {})["ci_80"] = [
            float(baseline + float(peak_ci80[0])),
            float(baseline + float(peak_ci80[1])),
        ]
    if peak_ci is not None and len(peak_ci) == 2:
        out_intervals.setdefault("peak_glucose_abs", {})["ci_95"] = [
            float(baseline + float(peak_ci[0])),
            float(baseline + float(peak_ci[1])),
        ]
    return out_preds, out_intervals


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


@app.get("/backend", response_class=HTMLResponse)
def backend_page() -> str:
    html_path = FRONTEND_DIR / "backend.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/v1/healthz")
def healthz() -> Dict[str, str]:
    return {
        "status": "ok",
        "model_root": str(model_root),
        "model_family": str(model_family),
        "active_model_id": str(ACTIVE_MODEL_ID),
        "available_models": ",".join(sorted(MODEL_REGISTRY.keys())),
    }


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
        preds, intervals, ci_metadata = service.predict_with_intervals(features)
        preds_out, intervals_out = _augment_predictions_for_output(preds=preds, intervals=intervals, features=features)
        meal_type = str(features.get("meal_type", "Lunch"))
        cohort = _build_cohort_comparison(meal_type=meal_type, preds=preds)
        personal = _build_personal_comparison(user_id="example-user", meal_type=meal_type, preds=preds)
        analysis_text = _build_analysis_text(
            features=features, preds=preds, intervals=intervals, cohort=cohort, personal=personal, ci_metadata=ci_metadata
        )
        return {
            "predictions": preds_out,
            "prediction_intervals": intervals_out,
            "ci_metadata": ci_metadata,
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
    model_options = [
        {
            "model_id": k,
            "label": _model_display_name(k, str(v.get("model_family", ""))),
            "model_root": str(v.get("model_root")),
        }
        for k, v in sorted(MODEL_REGISTRY.items(), key=lambda kv: kv[0])
    ]
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
        "default_model_id": ACTIVE_MODEL_ID,
        "model_options": model_options,
    }


@app.post("/v1/build-features-from-raw", response_model=RawFeatureBuildResponse)
def build_features_from_raw(req: RawFeatureBuildRequest) -> RawFeatureBuildResponse:
    return RawFeatureBuildResponse(features=_build_required_features_or_422(req))


@app.post("/v1/diagnostics/raw-input")
def diagnostics_raw_input(req: RawFeatureBuildRequest) -> Dict[str, Any]:
    features = _build_required_features_or_422(req)
    preds, intervals, ci_metadata = service.predict_with_intervals(features)
    preds_out, intervals_out = _augment_predictions_for_output(preds=preds, intervals=intervals, features=features)
    return {
        "raw_contract": {
            "strict_required_fields": True,
            "default_fill_for_required_raw_fields": False,
        },
        "features_used_for_model": features,
        "predictions": preds_out,
        "prediction_intervals": intervals_out,
        "ci_metadata": ci_metadata,
    }


@app.post("/v1/diagnostics/numeric-zscores")
def diagnostics_numeric_zscores(req: RawFeatureBuildRequest) -> Dict[str, Any]:
    features = _build_required_features_or_422(req)
    return {
        "features_used_for_model": features,
        "numeric_zscores": service.numeric_zscores(features),
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
    model_bundle = _resolve_model_bundle(req.model_id)
    run_service: ScalarMLPInferenceService = model_bundle["service"]
    run_model_id = str(model_bundle["model_id"])
    run_model_family = str(model_bundle["model_family"])

    try:
        preds, intervals, ci_metadata = run_service.predict_with_intervals(features)
    except Exception as exc:
        _append_log(
            {
                "ts_utc": _utc_now(),
                "request_id": request_id,
                "status": "error",
                "error": str(exc),
                "model_id": run_model_id,
                "model_family": run_model_family,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        )
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    meal_type = str(features.get("meal_type", "Lunch"))
    cohort = _build_cohort_comparison(meal_type=meal_type, preds=preds)
    personal = _build_personal_comparison(user_id=req.user_id, meal_type=meal_type, preds=preds)
    analysis_text = _build_analysis_text(
        features=features, preds=preds, intervals=intervals, cohort=cohort, personal=personal, ci_metadata=ci_metadata
    )
    preds_out, intervals_out = _augment_predictions_for_output(preds=preds, intervals=intervals, features=features)

    notes = [
        "This endpoint predicts scalar targets from provided meal/glucose context.",
        "Accepts a single user-facing raw payload (`meal_info` + `pre_glucose_series`) and computes engineered features server-side.",
        "Uncertainty bands use fold-ensemble sigma spread with OOF residual calibration when available.",
        "Cohort comparison uses CGMacros-derived same-meal-type OOF distributions with 30-bin histograms.",
        "Personal comparison includes user-history percentiles and 30-bin histograms when enough history is available.",
        "AUC-abs and peak-amplitude predictions are constrained to nonnegative values in postprocessing.",
        "For interpretability, response includes `peak_glucose_abs` (baseline + peak_amplitude).",
        "Use for research workflows; not for diagnosis or treatment decisions.",
    ]

    expected = run_service.expected_columns() if req.include_expected_columns else None
    resp = PredictResponse(
        request_id=request_id,
        predictions=preds_out,
        prediction_intervals=intervals_out,
        ci_metadata=ci_metadata,
        cohort_comparison=cohort,
        personal_comparison=personal,
        analysis_text=analysis_text,
        model_family=run_model_family,
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
            "predictions": preds_out,
            "prediction_intervals": intervals_out,
            "features": {"meal_type": meal_type},
            "user_id": req.user_id,
            "model_id": run_model_id,
            "model_family": run_model_family,
            "ci_method": str(ci_metadata.get("method", "unknown")),
            "ci_metadata": ci_metadata,
        }
    )
    return resp


@app.post("/v1/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    if not req.items:
        raise HTTPException(status_code=422, detail="`items` must contain at least one feature object.")
    model_bundle = _resolve_model_bundle(req.model_id)
    run_service: ScalarMLPInferenceService = model_bundle["service"]
    run_model_id = str(model_bundle["model_id"])
    run_model_family = str(model_bundle["model_family"])

    preds_all = []
    try:
        for item in req.items:
            features = _build_required_features_or_422(item)
            preds = run_service.predict_one(features)
            preds_out, _ = _augment_predictions_for_output(preds=preds, intervals={}, features=features)
            preds_all.append(preds_out)
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
                "model_id": run_model_id,
                "model_family": run_model_family,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        )
        raise HTTPException(status_code=400, detail=f"Batch inference failed: {exc}") from exc

    notes = [
        "Batch endpoint predicts scalar targets for each raw input object in `items`.",
        "Each item requires `meal_info` + `pre_glucose_series`; engineered features are computed server-side.",
        "Use for research workflows; not for diagnosis or treatment decisions.",
    ]
    expected = run_service.expected_columns() if req.include_expected_columns else None
    resp = PredictBatchResponse(
        request_id=request_id,
        model_family=run_model_family,
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
            "model_id": run_model_id,
            "model_family": run_model_family,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
    )
    return resp
