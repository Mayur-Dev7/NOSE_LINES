#!/usr/bin/env bash
# ==============================================================================
# One-Click Launcher for Cattle Muzzle AI Detector & Cropper
# Runs FastAPI Backend (Port 8000) & React Vite Frontend (Port 5173)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python environment not found at $VENV_PYTHON" >&2
    exit 1
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory not found at $FRONTEND_DIR" >&2
    exit 1
fi

echo "========================================================================"
echo " Starting Cattle Muzzle AI Detection Pipeline & React Web UI..."
echo "========================================================================"

# Free up ports 8000 and 5173 if currently occupied
cleanup_existing() {
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 5173/tcp 2>/dev/null || true
    sleep 1
}

# Trap to cleanly terminate children on Ctrl+C or kill
cleanup_on_exit() {
    echo ""
    echo "Stopping servers..."
    if [ -n "$BACKEND_PID" ]; then kill -TERM "$BACKEND_PID" 2>/dev/null || true; fi
    if [ -n "$FRONTEND_PID" ]; then kill -TERM "$FRONTEND_PID" 2>/dev/null || true; fi
    exit 0
}
trap cleanup_on_exit SIGINT SIGTERM EXIT

cleanup_existing

# 1. Start FastAPI Backend Server
echo "[1/3] Starting FastAPI Backend on port 8000..."
"$VENV_PYTHON" "${SCRIPT_DIR}/server.py" > "${SCRIPT_DIR}/backend.log" 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo -n "      Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:8000/api/status" >/dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    sleep 0.5
    echo -n "."
done

# 2. Start Vite React Frontend
echo "[2/3] Starting React Vite Frontend on port 5173..."
(
    cd "$FRONTEND_DIR"
    npm run dev -- --host 0.0.0.0 --port 5173
) > "${SCRIPT_DIR}/frontend.log" 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -n "      Waiting for frontend to initialize..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:5173/" >/dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    sleep 0.5
    echo -n "."
done

# 3. Detect Primary Local Network IP
LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}')
if [ -z "$LAN_IP" ]; then
    LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "$LAN_IP" ]; then
    LAN_IP="192.168.0.100"
fi

echo ""
echo "========================================================================"
echo "  Cow Muzzle AI Detection System is RUNNING!"
echo "========================================================================"
echo "  Hardware Engine: NVIDIA RTX 3050 Laptop GPU & AMD Ryzen 5 5600H"
echo "  Weights:         crop11dec.h5 (ResNet-101 Matterport Mask R-CNN)"
echo "  Backend API:     http://localhost:8000"
echo "  Local UI:        http://localhost:5173"
echo "  Network UI:      http://${LAN_IP}:5173"
echo "========================================================================"
echo ""
echo "http://${LAN_IP}:5173"
echo ""

# Keep running and wait for background jobs
while kill -0 "$BACKEND_PID" 2>/dev/null || kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 2
done
