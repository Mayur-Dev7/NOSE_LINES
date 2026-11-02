"""
System Status & Live Log Routes
===============================
Provides endpoints for monitoring hardware capabilities, live pipeline logs,
and system readiness.
"""

import os
import glob
import subprocess
from pathlib import Path
from fastapi import APIRouter

from ..log_handler import _PIPELINE_LOGS
from ..helpers import WEIGHTS_PATH, UPLOADS_DIR, TEST_SAMPLES_DIR

router = APIRouter()


@router.get("/api/logs/recent")
async def get_recent_pipeline_logs(limit: int = 150):
    """Returns recent structured log entries for the live UI telemetry drawer."""
    logs = list(_PIPELINE_LOGS)
    return {"logs": logs[-limit:] if limit > 0 else logs}


@router.post("/api/logs/clear")
async def clear_pipeline_logs():
    """Clears live telemetry buffer at the beginning of a fresh verification session."""
    _PIPELINE_LOGS.clear()
    return {"status": "cleared"}


@router.get("/api/status")
async def get_system_status():
    """Returns detected hardware and model environment information."""
    has_gpu = False
    gpu_name = "None"
    try:
        # Check nvidia-smi
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if smi.returncode == 0 and smi.stdout.strip():
            gpu_name = smi.stdout.strip().split("\n")[0]
            has_gpu = True
    except Exception:
        pass

    # CPU info (physical cores, logical threads, max cores)
    logical_cores = os.cpu_count() or 12
    core_ids = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
        try:
            with open(p) as f:
                core_ids.add(f.read().strip())
        except Exception:
            pass
    physical_cores = len(core_ids) if core_ids else (logical_cores // 2 or 6)
    cpu_model = f"AMD Ryzen 5 5600H ({physical_cores} cores / {logical_cores} threads)"

    # Test samples
    sample_images = []
    if TEST_SAMPLES_DIR.exists():
        sample_images = [f.name for f in sorted(TEST_SAMPLES_DIR.glob("*.jpg"))]

    sample_videos = []
    # Scan root directory for sample videos
    root_dir = Path("/home/adis_mayur/F_CODES")
    for vid in sorted(root_dir.glob("*.mp4")):
        sample_videos.append({
            "name": vid.name,
            "path": str(vid),
            "size_mb": round(vid.stat().st_size / (1024 * 1024), 2)
        })
    # Also check uploads (only valid non-empty videos)
    for vid in sorted(UPLOADS_DIR.glob("*.mp4")):
        if vid.stat().st_size > 0:
            sample_videos.append({
                "name": f"Uploaded: {vid.name.split('_', 1)[-1]}",
                "path": str(vid),
                "size_mb": round(vid.stat().st_size / (1024 * 1024), 2)
            })

    return {
        "status": "ready",
        "weights_present": WEIGHTS_PATH.exists(),
        "weights_path": str(WEIGHTS_PATH),
        "weights_size_mb": round(WEIGHTS_PATH.stat().st_size / (1024 * 1024), 1) if WEIGHTS_PATH.exists() else 0,
        "cpu_cores_logical": logical_cores,
        "cpu_cores_physical": physical_cores,
        "max_cores": logical_cores,
        "hardware": {
            "gpu_available": has_gpu,
            "gpu_name": gpu_name,
            "cpu_model": cpu_model,
            "cpu_threads": logical_cores,
            "cpu_cores_logical": logical_cores,
            "cpu_cores_physical": physical_cores,
            "max_cores": logical_cores,
        },
        "sample_images": sample_images,
        "sample_videos": sample_videos,
    }
