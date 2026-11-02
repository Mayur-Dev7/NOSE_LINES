"""
FastAPI Application Initialization & Configuration
==================================================
Initializes the FastAPI application, configures CORS middleware,
preloads CUDA runtime libraries, configures CPU thread hygiene,
registers modular API routes, and mounts static file directories.
"""

import os
import sys
import glob

# Prevent muzzle_cropper from re-executing server process via execve
os.environ["MRCNN_NVIDIA_CONFIGURED"] = "1"
# Enable oneDNN optimizations for accelerated CPU inference
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

from .helpers import (
    BASE_DIR,
    OUTPUT_DIR,
    TEST_SAMPLES_DIR,
    FRONTEND_DIST,
)
from .log_handler import _PIPELINE_LOGS, PipelineLogHandler, logger

# Preload NVIDIA CUDA 11 runtime libraries into global process space so TensorFlow finds GPU:0
# IMPORTANT: Skip libnvblas.so — it intercepts CPU BLAS calls (sgemm) and routes them to GPU,
# which fails without nvblas.conf and produces NaN/Inf values crashing JSON serialization.
try:
    import ctypes
    site_packages = os.path.join(str(BASE_DIR), ".venv", "lib", "python3.8", "site-packages")
    for so_file in sorted(glob.glob(os.path.join(site_packages, "nvidia", "*", "lib", "*.so*"))):
        if "libnvblas" in os.path.basename(so_file):
            continue  # Skip NVBLAS to prevent CPU BLAS interception failures
        try:
            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass
except Exception:
    pass

import cv2

# Thread and runtime hygiene settings
try:
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(6)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import register_routes

app = FastAPI(title="Cattle Muzzle AI Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all modular route handlers
register_routes(app)

# Static file serving
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/test_samples", StaticFiles(directory=str(TEST_SAMPLES_DIR)), name="test_samples")

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
