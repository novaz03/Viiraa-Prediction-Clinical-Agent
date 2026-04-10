#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bsub/web/submit_viiraa_webapp_interactive.sh --port <PORT> [options]

Required:
  --port <PORT>        Host/container port to expose for the web app.

Optional:
  --queue <QUEUE>      LSF queue (default: artsci-interactive)
  --wall <HH:MM>       Walltime (default: 08:00)
  --cores <N>          CPU cores (default: 4)
  --mem-gb <N>         Memory in GB (default: 64)
  --shadow <PATH>      Shadow root (default: /storage1/fs1/nlin/Active/sizhe_shadow_min)
  --image <IMAGE>      Docker image (default: sizhez03/pytorch_cuda:0.0.2)

This submits an interactive job and starts:
  python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port <PORT>
EOF
}

PORT=""
QUEUE="artsci-interactive"
WALL="08:00"
CORES="4"
MEM_GB="64"
SHADOW="/storage1/fs1/nlin/Active/sizhe_shadow_min"
IMAGE="sizhez03/pytorch_cuda:0.0.2"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --queue)
      QUEUE="${2:-}"
      shift 2
      ;;
    --wall)
      WALL="${2:-}"
      shift 2
      ;;
    --cores)
      CORES="${2:-}"
      shift 2
      ;;
    --mem-gb)
      MEM_GB="${2:-}"
      shift 2
      ;;
    --shadow)
      SHADOW="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "${PORT}" ]; then
  echo "--port is required." >&2
  usage
  exit 1
fi
if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "--port must be an integer." >&2
  exit 1
fi
if [ "${PORT}" -lt 1024 ] || [ "${PORT}" -gt 65535 ]; then
  echo "--port must be in [1024, 65535]." >&2
  exit 1
fi

REPO_ROOT="/storage1/fs1/nlin/Active/sizhe_shadow_min/scratch1/fs1/nlin/sizhe/Viiraa/Viiraa-Prediction-Clinical-Agent"
if [ ! -d "${REPO_ROOT}" ]; then
  REPO_ROOT="$(pwd)"
fi

PRED_REPO="${REPO_ROOT%/Viiraa-Prediction-Clinical-Agent}/Viiraa-Prediction"

export SHADOW
export LSF_DOCKER_VOLUMES="$SHADOW/scratch1:/scratch1 /home/sizhe:/home/sizhe /storage1/fs1/nlin/Active/sizhe:/storage1/fs1/nlin/Active/sizhe $SHADOW:/storage1/fs1/nlin/Active/sizhe_shadow_min"
export LSF_DOCKER_SHM_SIZE=16g
export LSF_DOCKER_PORTS="${PORT}:${PORT}"
export VIIRAA_WEBAPP_PORT="${PORT}"

echo "[interactive-webapp] repo=${REPO_ROOT}"
echo "[interactive-webapp] queue=${QUEUE}"
echo "[interactive-webapp] port=${PORT}"
echo "[interactive-webapp] image=${IMAGE}"
echo

read -r -d '' START_CMD <<EOF || true
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PRED_REPO}"
export SCALAR_MLP_MODEL_ROOT="${PRED_REPO}/outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models"
export SCALAR_MLP_EXAMPLE_PATH="${PRED_REPO}/examples/mlp_models/sample_input_single_meal.json"
HOST_SHORT=\$(hostname -s 2>/dev/null || hostname)
HOST_FQDN=\$(hostname -f 2>/dev/null || hostname)
echo "[webapp] host=\${HOST_FQDN}"
echo "[webapp] port=${PORT}"
echo "[webapp] tunnel: ssh -N -L ${PORT}:\${HOST_SHORT}:${PORT} <user>@<login-host>"
echo "[webapp] open:   http://localhost:${PORT}/"
exec /storage1/fs1/nlin/Active/sizhe/conda/envs/Viiraa/bin/python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port "${PORT}"
EOF

bsub -Is \
  -G compute-nlin \
  -q "${QUEUE}" \
  -R "select[port${PORT}=1] rusage[mem=${MEM_GB}GB]" \
  -a "docker(${IMAGE})" \
  -W "${WALL}" \
  -n "${CORES}" \
  /bin/bash -lc "${START_CMD}"
