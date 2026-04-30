from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from webapp.backend.model_arch import (
    build_model as _build_model_from_arch,
    inverse_transform_target as _inverse_transform_target,
    transform_categorical as _transform_categorical,
    transform_numeric as _transform_numeric,
)


TARGETS = ("auc_120_abs", "iauc_120", "peak_amplitude")
NONNEGATIVE_TARGETS = {"auc_120_abs", "peak_amplitude"}


@dataclass
class LoadedTargetModel:
    target: str
    ckpt: Dict[str, Any]
    model: torch.nn.Module
    numeric_cols: List[str]
    categorical_cols: List[str]
    feature_names: List[str]
    keep_idx: List[int]
    full_feature_names: List[str]


def _build_model(arch: str, input_dim: int, hidden_dims: List[int], dropout: float) -> torch.nn.Module:
    return _build_model_from_arch(arch=arch, input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


def _derive_features(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    carbs = float(out.get("carbs_g", 0.0) or 0.0)
    protein = float(out.get("protein_g", 0.0) or 0.0)
    fat = float(out.get("fat_g", 0.0) or 0.0)
    meal_cals = float(out.get("meal_calories", 0.0) or 0.0)
    mins_last = float(out.get("minutes_since_last_meal", 0.0) or 0.0)

    if "log1p_carbs_g" not in out:
        out["log1p_carbs_g"] = float(np.log1p(max(carbs, 0.0)))
    if "log1p_protein_g" not in out:
        out["log1p_protein_g"] = float(np.log1p(max(protein, 0.0)))
    if "log1p_fat_g" not in out:
        out["log1p_fat_g"] = float(np.log1p(max(fat, 0.0)))
    if "log1p_meal_calories" not in out:
        out["log1p_meal_calories"] = float(np.log1p(max(meal_cals, 0.0)))
    if "log1p_minutes_since_last_meal" not in out:
        out["log1p_minutes_since_last_meal"] = float(np.log1p(max(mins_last, 0.0)))

    macro_cals = carbs * 4.0 + protein * 4.0 + fat * 9.0
    if macro_cals > 0.0:
        # Training expects macro-calorie fractions in [0, 1], not percentages.
        if "pct_macro_cals_carbs" not in out:
            out["pct_macro_cals_carbs"] = carbs * 4.0 / macro_cals
        if "pct_macro_cals_protein" not in out:
            out["pct_macro_cals_protein"] = protein * 4.0 / macro_cals
        if "pct_macro_cals_fat" not in out:
            out["pct_macro_cals_fat"] = fat * 9.0 / macro_cals

    if "interaction_mins_since_last_x_carbs" not in out:
        out["interaction_mins_since_last_x_carbs"] = mins_last * carbs

    meal_type = str(out.get("meal_type", "") or "")
    meal_lower = meal_type.lower()
    is_breakfast = 1.0 if "breakfast" in meal_lower else 0.0
    is_lunch = 1.0 if "lunch" in meal_lower else 0.0
    is_dinner = 1.0 if "dinner" in meal_lower else 0.0
    is_snack = 1.0 if meal_lower in {"snack", "snacks"} or "snack" in meal_lower else 0.0
    out.setdefault("interaction_carbs_x_meal_breakfast", carbs * is_breakfast)
    out.setdefault("interaction_carbs_x_meal_lunch", carbs * is_lunch)
    out.setdefault("interaction_carbs_x_meal_dinner", carbs * is_dinner)
    out.setdefault("interaction_carbs_x_meal_snack", carbs * is_snack)
    out.setdefault("interaction_carbs_x_meal_snacks", carbs * is_snack)
    return out


class ScalarMLPInferenceService:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.models: Dict[str, LoadedTargetModel] = {}
        self.fold_models: Dict[str, List[LoadedTargetModel]] = {}
        self.residual_calibration: Dict[str, Dict[str, float]] = {}
        self._load()

    @staticmethod
    def _load_target_ckpt(target: str, ckpt: Dict[str, Any]) -> LoadedTargetModel:
        model = _build_model(
            arch=str(ckpt["model_arch"]),
            input_dim=int(ckpt["input_dim"]),
            hidden_dims=list(ckpt["hidden_dims"]),
            dropout=float(ckpt["dropout"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        numeric_cols = list(ckpt["numeric_cols"])
        categorical_cols = list(ckpt["categorical_cols"])
        pre = ckpt["preprocess"]
        full_feature_names = [f"num::{c}" for c in numeric_cols]
        for c in categorical_cols:
            full_feature_names.extend([f"cat::{c}::{lvl}" for lvl in pre["cat_levels"][c]])
        idx_map = {name: i for i, name in enumerate(full_feature_names)}
        feature_names = list(pre["feature_names"])
        keep_idx = [idx_map[name] for name in feature_names]

        return LoadedTargetModel(
            target=target,
            ckpt=ckpt,
            model=model,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            feature_names=feature_names,
            keep_idx=keep_idx,
            full_feature_names=full_feature_names,
        )

    def _load(self) -> None:
        for target in TARGETS:
            ckpt_path = self.model_root / target / "model.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.models[target] = self._load_target_ckpt(target, ckpt)

        fold_root = self.model_root.parent / "best_fold_model"
        for target in TARGETS:
            self.fold_models[target] = []
            target_fold_dir = fold_root / target
            if not target_fold_dir.exists():
                continue
            for fold_dir in sorted(target_fold_dir.glob("fold_*")):
                ckpt_path = fold_dir / "model.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self.fold_models[target].append(self._load_target_ckpt(target, ckpt))
        self._load_residual_calibration()

    def _load_residual_calibration(self) -> None:
        out: Dict[str, Dict[str, float]] = {}
        fold_root = self.model_root.parent / "best_fold_model"
        for target in TARGETS:
            p = fold_root / target / "oof_predictions.parquet"
            if not p.exists():
                continue
            true_col = f"true__{target}"
            pred_col = f"pred__{target}"
            try:
                df = pd.read_parquet(p, columns=[true_col, pred_col])
            except Exception:
                continue
            if true_col not in df.columns or pred_col not in df.columns:
                continue
            y_true = pd.to_numeric(df[true_col], errors="coerce")
            y_pred = pd.to_numeric(df[pred_col], errors="coerce")
            mask = y_true.notna() & y_pred.notna()
            if not bool(mask.any()):
                continue
            resid_abs = np.abs(y_true[mask].to_numpy(dtype=np.float64) - y_pred[mask].to_numpy(dtype=np.float64))
            out[target] = {
                "n": float(resid_abs.size),
                "abs_q80": float(np.quantile(resid_abs, 0.80)),
                "abs_q95": float(np.quantile(resid_abs, 0.95)),
            }
        self.residual_calibration = out

    def expected_columns(self) -> Dict[str, List[str]]:
        cols: Dict[str, set[str]] = {t: set() for t in TARGETS}
        for t, loaded in self.models.items():
            cols[t].update(loaded.numeric_cols)
            cols[t].update(loaded.categorical_cols)
        union = sorted(set().union(*cols.values()))
        return {"union": union, **{t: sorted(list(v)) for t, v in cols.items()}}

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, float]:
        row = _derive_features(features)
        out: Dict[str, float] = {}
        for target, loaded in self.models.items():
            out[target] = self._predict_with_loaded(row, loaded)
        return out

    @staticmethod
    def _predict_with_loaded(row: Dict[str, Any], loaded: LoadedTargetModel) -> float:
        pre = loaded.ckpt["preprocess"]
        df = pd.DataFrame([row])

        for col in loaded.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in loaded.categorical_cols:
            if col not in df.columns:
                df[col] = "__MISSING__"

        x_num = _transform_numeric(
            df,
            loaded.numeric_cols,
            np.asarray(pre["num_medians"], dtype=np.float32),
            np.asarray(pre["num_means"], dtype=np.float32),
            np.asarray(pre["num_stds"], dtype=np.float32),
        )
        x_cat = _transform_categorical(df, loaded.categorical_cols, pre["cat_levels"])
        x_full = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)
        x = x_full[:, loaded.keep_idx]

        with torch.no_grad():
            pred = loaded.model(torch.from_numpy(x)).detach().cpu().numpy().astype(np.float64)
        pred = _inverse_transform_target(pred, loaded.ckpt["target_preprocess"])
        return float(pred[0])

    def predict_with_intervals(
        self, features: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]], Dict[str, Any]]:
        row = _derive_features(features)
        preds: Dict[str, float] = {}
        intervals: Dict[str, Dict[str, List[float]]] = {}
        method = "heuristic_spread_v1"
        ci_meta: Dict[str, Any] = {"method": method, "targets": {}}
        methods_seen: set[str] = set()
        for target, loaded in self.models.items():
            folds = self.fold_models.get(target, [])
            fold_preds = [self._predict_with_loaded(row, fm) for fm in folds]
            fold_sigma: float | None = None
            if len(fold_preds) >= 2:
                arr = np.asarray(fold_preds, dtype=np.float64)
                p = float(np.median(arr))
                # Sigma-based spread from best-fold ensemble predictions.
                sd = float(np.std(arr, ddof=1))
                fold_sigma = sd
                hw80 = 1.2816 * sd
                hw95 = 1.9600 * sd
                ci80 = [float(p - hw80), float(p + hw80)]
                ci95 = [float(p - hw95), float(p + hw95)]
                method = "fold_ensemble_sigma_v1"
            else:
                p = self._predict_with_loaded(row, loaded)
                # Fallback when fold models are unavailable.
                sd = max(1e-6, 0.06 * max(abs(p), 1.0))
                fold_sigma = None
                hw80 = 1.2816 * sd
                hw95 = 1.9600 * sd
                ci80 = [float(p - hw80), float(p + hw80)]
                ci95 = [float(p - hw95), float(p + hw95)]

            calib = self.residual_calibration.get(target)
            if calib is not None:
                q80 = float(calib.get("abs_q80", 0.0))
                q95 = float(calib.get("abs_q95", 0.0))
                ci80 = [min(float(ci80[0]), float(p - q80)), max(float(ci80[1]), float(p + q80))]
                ci95 = [min(float(ci95[0]), float(p - q95)), max(float(ci95[1]), float(p + q95))]
                if method.startswith("fold_ensemble"):
                    method = "fold_ensemble_sigma_plus_oof_resid_v1"
                else:
                    method = "heuristic_plus_oof_resid_v2"

            if target in NONNEGATIVE_TARGETS:
                p = max(0.0, float(p))
                ci80 = [max(0.0, float(ci80[0])), max(0.0, float(ci80[1]))]
                ci95 = [max(0.0, float(ci95[0])), max(0.0, float(ci95[1]))]

            ci80 = [min(ci80[0], p), max(ci80[1], p)]
            ci95 = [min(ci95[0], p), max(ci95[1], p)]
            preds[target] = p

            intervals[target] = {
                "ci_80": [float(ci80[0]), float(ci80[1])],
                "ci_95": [float(ci95[0]), float(ci95[1])],
            }
            ci_meta["targets"][target] = {
                "fold_count": int(len(fold_preds)),
                "fold_sigma": float(fold_sigma) if fold_sigma is not None else None,
                "calibration_samples": int(calib["n"]) if calib is not None else 0,
                "residual_abs_q80": float(calib["abs_q80"]) if calib is not None else None,
                "residual_abs_q95": float(calib["abs_q95"]) if calib is not None else None,
            }
            methods_seen.add(method)
        ci_meta["method"] = method if len(methods_seen) == 1 else "mixed"
        return preds, intervals, ci_meta

    def demographic_fields(self) -> List[str]:
        cols = set(self.expected_columns()["union"])
        keys = ("age", "sex", "gender", "height", "weight", "bmi", "race", "ethnicity")
        out = [c for c in sorted(cols) if any(k in c.lower() for k in keys)]
        return out

    def numeric_zscores(self, features: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        row = _derive_features(features)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for target, loaded in self.models.items():
            pre = loaded.ckpt["preprocess"]
            means = np.asarray(pre["num_means"], dtype=np.float64)
            stds = np.asarray(pre["num_stds"], dtype=np.float64)
            medians = np.asarray(pre["num_medians"], dtype=np.float64)
            rows: List[Dict[str, Any]] = []
            for i, col in enumerate(loaded.numeric_cols):
                raw_val = row.get(col, np.nan)
                imputed = False
                try:
                    val = float(raw_val)
                except Exception:
                    val = float("nan")
                if not np.isfinite(val):
                    val = float(medians[i])
                    imputed = True
                denom = float(stds[i]) if abs(float(stds[i])) > 1e-12 else 1.0
                z = (float(val) - float(means[i])) / denom
                rows.append(
                    {
                        "feature": col,
                        "value": float(val),
                        "z": float(z),
                        "imputed": bool(imputed),
                    }
                )
            out[target] = rows
        return out

    @staticmethod
    def load_example(example_path: Path) -> Dict[str, Any]:
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unexpected example payload in {example_path}")
