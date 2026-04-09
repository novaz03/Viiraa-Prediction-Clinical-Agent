#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bsub/web/submit_viiraa_webapp.sh --port <PORT> [--queue artsci] [--shadow /path]

Required:
  --port <PORT>      Host/container port to expose for the web app.

Optional:
  --queue <QUEUE>    LSF queue (default: artsci).
  --shadow <PATH>    Shadow root (default: /storage1/fs1/nlin/Active/sizhe_shadow_min).

This script exports the required Docker variables before submitting the web app BSUB job.
EOF
}

PORT=""
QUEUE="artsci"
SHADOW="/storage1/fs1/nlin/Active/sizhe_shadow_min"

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
    --shadow)
      SHADOW="${2:-}"
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

cd "${REPO_ROOT}"

export SHADOW
export LSF_DOCKER_VOLUMES="$SHADOW/scratch1:/scratch1 /home/sizhe:/home/sizhe /storage1/fs1/nlin/Active/sizhe:/storage1/fs1/nlin/Active/sizhe $SHADOW:/storage1/fs1/nlin/Active/sizhe_shadow_min"
export LSF_DOCKER_SHM_SIZE=16g
export LSF_DOCKER_PORTS="${PORT}:${PORT}"
export VIIRAA_WEBAPP_PORT="${PORT}"

echo "[submit-webapp] repo=${REPO_ROOT}"
echo "[submit-webapp] queue=${QUEUE}"
echo "[submit-webapp] port=${PORT}"
echo "[submit-webapp] SHADOW=${SHADOW}"
echo "[submit-webapp] LSF_DOCKER_PORTS=${LSF_DOCKER_PORTS}"

bsub -q "${QUEUE}" < scripts/bsub/web/viiraa_webapp_api.bsub
