#!/usr/bin/env bash
# ==============================================================================
# Helper Runner Script for Cow Muzzle Detection & Cropping Pipeline
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment python not found at $VENV_PYTHON" >&2
    exit 1
fi

exec "$VENV_PYTHON" "${SCRIPT_DIR}/muzzle_cropper.py" "$@"
