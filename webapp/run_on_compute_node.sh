#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash webapp/run_on_compute_node.sh [PORT]
#
# Starts the web API on the current node and prints tunnel instructions.

PORT="${1:-8000}"
HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"

echo "Starting Viiraa web app on node: ${HOSTNAME_FQDN}"
echo "Port: ${PORT}"
echo
echo "From your laptop, create SSH tunnel (edit login host/user as needed):"
echo "  ssh -N -L ${PORT}:${HOSTNAME_SHORT}:${PORT} <user>@<login-host>"
echo
echo "Then open:"
echo "  http://localhost:${PORT}/"
echo
echo "If direct hop to compute host is not allowed, use ProxyJump:"
echo "  ssh -N -J <user>@<login-host> -L ${PORT}:localhost:${PORT} <user>@${HOSTNAME_SHORT}"
echo

uvicorn webapp.backend.app:app --host 0.0.0.0 --port "${PORT}"
