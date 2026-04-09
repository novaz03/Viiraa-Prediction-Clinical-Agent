#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RiskFlag:
    level: str
    code: str
    message: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_pred_delta(payload: Dict[str, Any]) -> List[float]:
    if isinstance(payload.get("pred_mean"), list):
        return [float(v) for v in payload["pred_mean"]]
    if isinstance(payload.get("pred_delta_glucose"), list):
        return [float(v) for v in payload["pred_delta_glucose"]]
    if isinstance(payload.get("delta_glucose"), list):
        return [float(v) for v in payload["delta_glucose"]]
    pred = payload.get("pred")
    quantiles = payload.get("quantiles")
    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        q_idx = 0
        if isinstance(quantiles, list) and quantiles:
            diffs = [abs(float(q) - 0.5) for q in quantiles]
            q_idx = int(min(range(len(diffs)), key=lambda i: diffs[i]))
        return [float(row[q_idx]) for row in pred]
    raise ValueError("Could not find a prediction series in input JSON.")


def _extract_minutes(payload: Dict[str, Any], n: int) -> List[int]:
    mins = payload.get("minutes")
    if isinstance(mins, list) and len(mins) == n:
        return [int(m) for m in mins]
    return list(range(1, n + 1))


def _compute_metrics(minutes: List[int], delta: List[float], baseline_glucose: Optional[float]) -> Dict[str, Any]:
    peak_idx = max(range(len(delta)), key=lambda i: delta[i])
    nadir_idx = min(range(len(delta)), key=lambda i: delta[i])

    out: Dict[str, Any] = {
        "peak_delta_mgdl": float(delta[peak_idx]),
        "peak_time_min": int(minutes[peak_idx]),
        "nadir_delta_mgdl": float(delta[nadir_idx]),
        "nadir_time_min": int(minutes[nadir_idx]),
        "delta_auc_sum": float(sum(delta)),
        "iauc_positive_sum": float(sum(max(v, 0.0) for v in delta)),
        "iauc_negative_sum": float(sum(min(v, 0.0) for v in delta)),
    }

    first_30_idx = [i for i, m in enumerate(minutes) if m <= 30]
    if first_30_idx:
        out["rise_in_first_30m_mgdl"] = float(max(delta[i] for i in first_30_idx) - delta[first_30_idx[0]])
    else:
        out["rise_in_first_30m_mgdl"] = 0.0

    if baseline_glucose is not None:
        abs_series = [baseline_glucose + v for v in delta]
        out["baseline_glucose_mgdl"] = float(baseline_glucose)
        out["peak_absolute_glucose_mgdl"] = float(max(abs_series))
        out["nadir_absolute_glucose_mgdl"] = float(min(abs_series))
        out["minutes_ge_180"] = int(sum(1 for v in abs_series if v >= 180.0))
        out["minutes_ge_250"] = int(sum(1 for v in abs_series if v >= 250.0))
        out["minutes_le_70"] = int(sum(1 for v in abs_series if v <= 70.0))
        out["minutes_le_54"] = int(sum(1 for v in abs_series if v <= 54.0))
    return out


def _risk_flags(metrics: Dict[str, Any]) -> List[RiskFlag]:
    flags: List[RiskFlag] = []
    peak_delta = float(metrics["peak_delta_mgdl"])
    rise_30 = float(metrics["rise_in_first_30m_mgdl"])

    if peak_delta >= 60:
        flags.append(RiskFlag("moderate", "large_excursion", "Predicted large post-meal glucose excursion (delta >= 60 mg/dL)."))
    elif peak_delta >= 40:
        flags.append(RiskFlag("low", "moderate_excursion", "Predicted moderate post-meal glucose excursion (delta >= 40 mg/dL)."))

    if rise_30 >= 30:
        flags.append(RiskFlag("low", "rapid_rise", "Predicted rapid early rise in first 30 minutes (>= 30 mg/dL)."))

    if "peak_absolute_glucose_mgdl" in metrics:
        if metrics["minutes_ge_250"] > 0:
            flags.append(RiskFlag("high", "severe_hyperglycemia_window", "Predicted glucose reaches >= 250 mg/dL; discuss with clinician if persistent."))
        elif metrics["minutes_ge_180"] >= 30:
            flags.append(RiskFlag("moderate", "prolonged_hyperglycemia_window", "Predicted glucose >= 180 mg/dL for >= 30 minutes."))
        elif metrics["minutes_ge_180"] > 0:
            flags.append(RiskFlag("low", "brief_hyperglycemia_window", "Predicted brief period above 180 mg/dL."))

        if metrics["minutes_le_54"] > 0:
            flags.append(RiskFlag("high", "severe_hypoglycemia_window", "Predicted glucose <= 54 mg/dL; requires prompt clinical review."))
        elif metrics["minutes_le_70"] > 0:
            flags.append(RiskFlag("moderate", "hypoglycemia_window", "Predicted glucose <= 70 mg/dL; watch for low-glucose symptoms and prevention."))
    return flags


def _suggestions(flags: List[RiskFlag]) -> List[str]:
    suggestions: List[str] = []
    codes = {f.code for f in flags}

    if "large_excursion" in codes or "moderate_excursion" in codes or "prolonged_hyperglycemia_window" in codes:
        suggestions.append("Consider reducing fast-absorbing carbohydrate load and pairing carbs with protein/fiber.")
        suggestions.append("Consider a short post-meal walk if clinically appropriate for the patient.")
    if "rapid_rise" in codes:
        suggestions.append("Prioritize lower glycemic-index choices and slower-carb meal composition.")
    if "hypoglycemia_window" in codes or "severe_hypoglycemia_window" in codes:
        suggestions.append("Review medication timing/dose with a clinician if low glucose risk repeats.")
        suggestions.append("Ensure a prevention plan for lows and reinforce symptom-based safety checks.")
    if not suggestions:
        suggestions.append("Current prediction does not show major acute risk flags; continue routine monitoring and individualized plan adherence.")

    suggestions.append("Use trend-level recommendations only; do not use this output alone for diagnosis or medication changes.")
    return suggestions


class ClinicalOutputChatAgent:
    def __init__(self, analysis: Dict[str, Any]) -> None:
        self.analysis = analysis

    def answer(self, question: str) -> str:
        q = question.strip().lower()
        m = self.analysis["metrics"]
        flags = self.analysis["risk_flags"]
        suggestions = self.analysis["suggestions"]

        if any(k in q for k in ["peak", "highest", "max"]):
            if "peak_absolute_glucose_mgdl" in m:
                return f"Predicted peak is {m['peak_absolute_glucose_mgdl']:.1f} mg/dL (delta {m['peak_delta_mgdl']:.1f}) at minute {m['peak_time_min']}."
            return f"Predicted peak delta is {m['peak_delta_mgdl']:.1f} mg/dL at minute {m['peak_time_min']}."

        if any(k in q for k in ["low", "hypo", "hypogly"]):
            if "nadir_absolute_glucose_mgdl" in m:
                return f"Predicted nadir is {m['nadir_absolute_glucose_mgdl']:.1f} mg/dL (delta {m['nadir_delta_mgdl']:.1f}) at minute {m['nadir_time_min']}."
            return f"Predicted nadir delta is {m['nadir_delta_mgdl']:.1f} mg/dL at minute {m['nadir_time_min']}."

        if "auc" in q:
            return f"delta AUC sum={m['delta_auc_sum']:.1f}, positive iAUC sum={m['iauc_positive_sum']:.1f}, negative iAUC sum={m['iauc_negative_sum']:.1f}."

        if any(k in q for k in ["suggest", "recommend", "what should", "next step"]):
            return "\n".join(f"- {s}" for s in suggestions)

        if "risk" in q or "flag" in q:
            if not flags:
                return "No major acute risk flags were detected from this trajectory."
            return "\n".join(f"- [{f['level']}] {f['message']}" for f in flags)

        return (
            "Summary:\n"
            f"- Peak delta {m['peak_delta_mgdl']:.1f} at {m['peak_time_min']} min\n"
            f"- Nadir delta {m['nadir_delta_mgdl']:.1f} at {m['nadir_time_min']} min\n"
            f"- Positive iAUC {m['iauc_positive_sum']:.1f}\n"
            f"- Risk flags: {len(flags)}\n"
            "Ask about peak, lows, auc, risk flags, or recommendations."
        )


def analyze_payload(payload: Dict[str, Any], baseline_glucose: Optional[float]) -> Dict[str, Any]:
    delta = _extract_pred_delta(payload)
    minutes = _extract_minutes(payload, len(delta))
    metrics = _compute_metrics(minutes, delta, baseline_glucose)
    flags = [f.__dict__ for f in _risk_flags(metrics)]
    suggestions = _suggestions([RiskFlag(**f) for f in flags])

    return {
        "schema_version": "v1",
        "safety_disclaimer": (
            "This tool provides model-output interpretation support only, not diagnosis or treatment advice. "
            "Clinical decisions require clinician review and patient-specific context."
        ),
        "metrics": metrics,
        "risk_flags": flags,
        "suggestions": suggestions,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic clinical-summary chat agent for glucose model output JSON.")
    ap.add_argument("--prediction-json", type=Path, required=True, help="Model output JSON path.")
    ap.add_argument("--baseline-glucose", type=float, default=None, help="Optional baseline glucose (mg/dL).")
    ap.add_argument("--report-out", type=Path, default=None, help="Optional path to write analysis JSON.")
    ap.add_argument("--question", type=str, default=None, help="One-shot question.")
    ap.add_argument("--chat", action="store_true", help="Interactive chat mode.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_json(args.prediction_json)
    analysis = analyze_payload(payload, args.baseline_glucose)
    agent = ClinicalOutputChatAgent(analysis)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"WROTE {args.report_out}")

    if args.question:
        print(agent.answer(args.question))
        return

    if args.chat:
        print("Clinical Output Chat Agent (type 'exit' to quit)")
        print(analysis["safety_disclaimer"])
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            print(agent.answer(q))
        return

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
