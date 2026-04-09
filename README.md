# Viiraa Prediction Clinical Chat Agent

Deterministic chat-style analyzer for glucose model output JSON.

## What it does
- Reads model output trajectories (`pred_mean`, `pred_delta_glucose`, `delta_glucose`, or quantile `pred`).
- Computes consistent summary metrics (peak, nadir, AUC/iAUC, early rise).
- Optionally converts deltas to absolute glucose with `--baseline-glucose`.
- Produces risk flags and suggestion text with safety guardrails.

## Safety
This tool is for model-output interpretation support only.
It is not diagnosis or treatment advice.

## Usage

```bash
python3 scripts/clinical_output_chat_agent.py \
  --prediction-json /path/to/frontend_output.json \
  --baseline-glucose 110 \
  --report-out report.json
```

One-shot question:

```bash
python3 scripts/clinical_output_chat_agent.py \
  --prediction-json /path/to/frontend_output.json \
  --baseline-glucose 110 \
  --question "What are the risk flags?"
```

Interactive chat:

```bash
python3 scripts/clinical_output_chat_agent.py \
  --prediction-json /path/to/frontend_output.json \
  --baseline-glucose 110 \
  --chat
```
