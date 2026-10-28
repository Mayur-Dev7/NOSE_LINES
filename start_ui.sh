#!/usr/bin/env bash
# ==============================================================================
# Cattle Muzzle AI Web UI Launcher
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

PORT="${1:-8000}"

echo "========================================================================"
echo " Starting Cattle Muzzle AI Web UI on http://localhost:${PORT}"
echo "========================================================================"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python venv not found at $VENV_PYTHON" >&2
    exit 1
fi

# Ensure frontend build exists
if [ ! -d "${SCRIPT_DIR}/frontend/dist" ]; then
    echo "Building React frontend..."
    (cd "${SCRIPT_DIR}/frontend" && npm run build)
fi

exec "$VENV_PYTHON" "${SCRIPT_DIR}/server.py"
