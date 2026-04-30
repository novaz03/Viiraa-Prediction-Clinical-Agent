# Web App Plan: Meal/Glucose Input -> AUC, iAUC, Peak Amplitude Prediction

## Objective
Build a website where users provide meal info and glucose context (using CGMacros-style examples), and receive predicted:
- `auc_120_abs`
- `iauc_120`
- `peak_amplitude`

Detailed clinical-style interpretation of outputs is explicitly planned as **Phase 2**.

---

## Scope and Phasing

## Phase 1 (Now): Prediction Website
- User input form for meal + glucose context.
- Backend inference service for the 3 scalar targets.
- Results page with raw predicted values and basic visualization.
- Input validation, reproducible examples, and logging.

## Phase 2 (Next): Detailed Output Analysis
- Structured interpretation layer (risk flags, pattern summaries, consistent suggestions).
- Rule-based and/or calibrated interpretation from predicted values.
- Expanded reports for longitudinal comparisons.

---

## Assumptions
- Existing trained scalar MLP artifacts remain the source of truth:
  - `outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models/*`
- Inputs follow the schema already demonstrated in repo examples:
  - `examples/mlp_models/sample_input_single_meal.csv`
  - `examples/mlp_models/sample_input_single_meal.json`
- Inference will run server-side in Python.

---

## Step-by-Step Development Plan

## Step 1: Lock the Input Contract
1. Define a strict request schema for prediction API using existing MLP example fields.
2. Separate fields into:
   - Required user-entered
   - Auto-derived (e.g., log/interactions if pipeline computes them)
   - Optional with defaults
3. Produce a single canonical JSON schema file (e.g., `webapp/schema/predict_request.schema.json`).

Deliverables:
- Request schema JSON
- Short docs table (`field`, `type`, `required`, `example`, `validation`)

## Step 2: Build Inference Adapter Layer
1. Create a standalone inference module that:
   - Loads all 3 target models once at process startup.
   - Validates and aligns input columns.
   - Applies the checkpoint preprocessing exactly.
   - Returns predictions in consistent units and precision.
2. Add deterministic test cases with known example input.

Suggested files:
- `webapp/backend/inference.py`
- `webapp/backend/model_registry.py`
- `webapp/tests/test_inference_contract.py`

Deliverables:
- `predict_targets(input_row) -> {auc_120_abs, iauc_120, peak_amplitude}`
- Unit tests passing

## Step 3: Build Backend API
1. Stand up a small API service (FastAPI recommended).
2. Add endpoints:
   - `POST /predict` (single meal)
   - `POST /predict/batch` (optional batch mode)
   - `GET /healthz`
   - `GET /example-input`
3. Enforce validation and structured error responses.
4. Add request/response logging (exclude sensitive data by default).

Suggested files:
- `webapp/backend/app.py`
- `webapp/backend/schemas.py`
- `webapp/backend/config.py`

Deliverables:
- Running local API with OpenAPI docs
- Error handling for malformed payloads

## Step 4: Build Frontend Website
1. Create a clean single-page app with:
   - Input form grouped by categories (meal, baseline glucose, premeal dynamics, lab context).
   - “Load CGMacros example” button.
   - Submit + loading state + inline validation.
2. Display predictions in:
   - Numeric cards
   - Simple chart (e.g., gauge/bars vs reference ranges)
3. Add “copy request JSON” and “download result JSON”.

Suggested files:
- `webapp/frontend/src/pages/PredictPage.tsx` (or equivalent)
- `webapp/frontend/src/components/InputForm.*`
- `webapp/frontend/src/components/PredictionCards.*`

Deliverables:
- Usable desktop/mobile UI
- End-to-end call to `/predict`

## Step 5: Add Example Data Pipeline
1. Convert one or more CGMacros examples into web-form-compatible payloads.
2. Store sanitized example payloads in repo.
3. Add frontend selector for multiple example scenarios (low/moderate/high excursion patterns).

Suggested files:
- `webapp/examples/example_low.json`
- `webapp/examples/example_moderate.json`
- `webapp/examples/example_high.json`

Deliverables:
- Reproducible demo scenarios
- Quick QA path

## Step 6: QA and Validation
1. Contract tests:
   - API accepts valid schema and rejects invalid payloads.
2. Numerical regression tests:
   - Predictions stable for fixed examples.
3. UX tests:
   - Required fields, loading state, error state, reset behavior.
4. Basic performance:
   - Target p95 latency under defined threshold (e.g., <300ms local single-request).

Deliverables:
- Test suite + baseline performance numbers

## Step 7: Packaging and Deployment
1. Containerize backend and frontend.
2. Define environment config:
   - Model path roots
   - CORS origins
   - Logging level
3. Add deployment docs:
   - Local dev
   - Staging
   - Production

Suggested files:
- `webapp/backend/Dockerfile`
- `webapp/frontend/Dockerfile`
- `webapp/docker-compose.yml`
- `webapp/DEPLOYMENT.md`

Deliverables:
- One-command local startup
- Staging-ready deployment docs

---

## Phase 2 Plan: Detailed Output Analysis Layer

## Step 8: Define Analysis Outputs
1. Specify analysis schema returned alongside predictions:
   - Derived indices (relative severity buckets, comparison vs user baseline)
   - Confidence/quality metadata
   - Risk flags and explanation snippets
2. Keep deterministic rule sets versioned.

## Step 9: Build Analysis Engine
1. Implement a standalone analysis module:
   - Input: scalar predictions + optional baseline/profile
   - Output: structured explanation object
2. Include hard safety guardrails:
   - Non-diagnostic disclaimers
   - Escalation conditions

## Step 10: UI for Detailed Reports
1. Add expandable analysis sections:
   - “What drove this output”
   - “Potential risk patterns”
   - “Action-oriented but non-prescriptive suggestions”
2. Add exportable report (JSON/PDF markdown).

---

## Technical Design Decisions (Recommended)
- Backend: FastAPI + Pydantic
- Frontend: React/Next.js
- Model Serving: in-process Python (single worker first; scale later)
- Validation: strict typed schema first, then transformation layer
- Versioning:
  - API version (`/v1/predict`)
  - Model version in response metadata
  - Analysis-rule version in Phase 2

---

## Risk Register and Mitigations
- Feature mismatch at inference:
  - Mitigation: schema lock + startup-time model feature audit.
- Silent preprocessing drift:
  - Mitigation: use checkpoint preprocessing only; disallow ad hoc transforms.
- User input quality variability:
  - Mitigation: constrained fields, defaults, and clear input hints.
- Clinical over-interpretation risk:
  - Mitigation: Phase 2 safety language and explicit non-diagnostic policy.

---

## Milestone Plan
- M1: Input contract + inference adapter (Steps 1-2)
- M2: Backend API productionized (Step 3)
- M3: Frontend MVP usable with examples (Steps 4-5)
- M4: QA + deployable package (Steps 6-7)
- M5: Detailed analysis engine + UI (Phase 2, Steps 8-10)

---

## Definition of Done (Phase 1)
- User can submit meal/glucose input through the site.
- Backend returns valid predictions for all 3 targets.
- UI shows predicted `auc_120_abs`, `iauc_120`, `peak_amplitude` clearly.
- Example payloads are available and reproducible.
- Tests cover schema, inference, and endpoint behavior.

## Definition of Done (Phase 2)
- Structured, consistent analysis is returned and displayed.
- Safety guardrails are present and tested.
- Reports are exportable and versioned.
