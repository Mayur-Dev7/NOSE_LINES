#!/usr/bin/env python3
"""
FastAPI Server for Cattle Muzzle Detection & Cropping UI
======================================================
Provides REST and Server-Sent Event (SSE) endpoints for:
- Hardware status & options (RTX 3050 GPU, Ryzen CPU 2 Cores vs 12 Threads)
- Video upload & frame extraction
- Real-time muzzle detection, segmentation & cropping with microsecond log streaming
- Hardware benchmark streaming
- Serving generated cropped muzzles and React frontend
"""

import os
import sys

# Prevent muzzle_cropper from re-executing server process via execve
os.environ["MRCNN_NVIDIA_CONFIGURED"] = "1"
import glob
import json
import time
import shutil
import asyncio
import functools
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict, Union
import logging

logger = logging.getLogger("server")

if not hasattr(asyncio, "to_thread"):
    async def _asyncio_to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_event_loop()
        func_call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func_call)
    asyncio.to_thread = _asyncio_to_thread

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from groove_extractor import extract_muzzle_grooves_pipeline
from batch_processor import process_image_batch
from groove_graph import extract_skeleton_graph, visualize_groove_graph
from template_consensus import build_consensus_template
from template_matcher import (
    save_reference_template,
    load_reference_template,
    list_saved_templates,
    compare_two_templates,
)
from template_exporter import (
    export_all_template_reports,
    export_consensus_summary_csv,
    export_per_image_measurements_csv,
    export_template_json,
)
from video_groove_extractor import (
    extract_five_frames_groove_features,
    compute_cosine_similarity,
)
from muzzle_matcher import (
    match_sift_ransac,
    match_minutiae,
    match_muzzle_frame_sets,
    match_two_videos_end_to_end,
    load_matcher_config,
    match_video_sets as match_spatial_video_sets,
)
WEIGHTS_PATH = Path("/home/adis_mayur/F_CODES/crop11dec.h5")
PYTHON_EXEC = BASE_DIR / ".venv" / "bin" / "python"
CROPPER_SCRIPT = BASE_DIR / "muzzle_cropper.py"
MATCHER_SCRIPT = BASE_DIR / "muzzle_matcher.py"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
TEST_SAMPLES_DIR = BASE_DIR / "test_samples"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = OUTPUT_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Cattle Muzzle AI Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
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
    # Also check uploads
    for vid in sorted(UPLOADS_DIR.glob("*.mp4")):
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a video or image file for muzzle cropping."""
    filename = Path(file.filename).name
    save_path = UPLOADS_DIR / f"{int(time.time())}_{filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_video = save_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    return {
        "success": True,
        "filename": filename,
        "path": str(save_path),
        "is_video": is_video,
        "size_mb": round(save_path.stat().st_size / (1024 * 1024), 2),
    }


def parse_cpu_cores(cpu_cores: Optional[Any]) -> Optional[int]:
    """Parses arbitrary user-specified CPU core counts, validating integer >= 1."""
    if cpu_cores is None:
        return None
    val_str = str(cpu_cores).strip()
    if not val_str or val_str.lower() in ("null", "none", "auto", "all"):
        return None
    try:
        val = int(val_str)
        return val if val >= 1 else None
    except (ValueError, TypeError):
        return None


def stream_subprocess_output(cmd: List[str], env_extra: Optional[dict] = None):
    """Generator streaming stdout/stderr as SSE events."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    try:
        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("[JSON_EVENT]"):
                json_part = line_str[len("[JSON_EVENT]"):].strip()
                try:
                    payload = json.loads(json_part)
                    p_type = payload.get("type", "")

                    if p_type == "matcher_log":
                        yield f"data: {json.dumps({'event': 'log', 'line': payload.get('text', '')})}\n\n"
                    elif p_type == "two_video_result":
                        res_data = payload.get("result", {})
                        # Convert paths to URLs
                        for key in ("video_a", "video_b"):
                            sub = res_data.get(key, {})
                            for pkey, ukey in [
                                ("crop_path", "crop_url"),
                                ("annotated_path", "annotated_url"),
                                ("enhanced_path", "enhanced_url"),
                                ("groove_response_path", "groove_response_url"),
                                ("groove_mask_path", "groove_mask_url"),
                                ("groove_skeleton_path", "groove_skeleton_url"),
                                ("groove_overlay_path", "groove_overlay_url"),
                            ]:
                                if pkey in sub and sub[pkey]:
                                    try:
                                        sub[ukey] = "/output/" + str(Path(sub[pkey]).relative_to(OUTPUT_DIR))
                                    except Exception:
                                        if ukey not in sub or not sub[ukey]:
                                            sub[ukey] = f"/output/{Path(sub[pkey]).name}"
                        m = res_data.get("match", {})
                        if "visual_path" in m and m["visual_path"]:
                            try:
                                m["visual_url"] = "/output/" + str(Path(m["visual_path"]).relative_to(OUTPUT_DIR))
                            except Exception:
                                m["visual_url"] = m["visual_path"]
                        yield f"data: {json.dumps({'event': 'two_video_result', 'data': res_data})}\n\n"
                    else:
                        # Convert absolute paths to API-relative URLs for frontend
                        if "annotated_path" in payload and payload["annotated_path"]:
                            try:
                                payload["annotated_url"] = "/output/" + str(Path(payload["annotated_path"]).relative_to(OUTPUT_DIR))
                            except Exception:
                                pass
                        if "crops" in payload:
                            for c in payload["crops"]:
                                if "crop_path" in c and c["crop_path"]:
                                    try:
                                        c["crop_url"] = "/output/" + str(Path(c["crop_path"]).relative_to(OUTPUT_DIR))
                                    except Exception:
                                        pass
                        yield f"data: {json.dumps({'event': 'detection', 'data': payload})}\n\n"
                except Exception as ex:
                    yield f"data: {json.dumps({'event': 'log', 'line': line_str})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'log', 'line': line_str})}\n\n"

        proc.stdout.close()
        returncode = proc.wait()
        yield f"data: {json.dumps({'event': 'done', 'returncode': returncode})}\n\n"
    except GeneratorExit:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception as e:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"


@app.get("/api/stream-video")
async def stream_video(
    video_path: str = Query(...),
    device: str = Query("auto"),
    cpu_cores: Optional[str] = Query(None),
    frame_interval: int = Query(5),
    max_frames: Optional[int] = Query(10),
    confidence: float = Query(0.70),
    square: bool = Query(True),
    resize: int = Query(224),
    fast_mode: Optional[bool] = Query(None),
):
    """Processes a video file and streams frame crops and logs over SSE."""
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    job_output_dir = OUTPUT_DIR / f"run_video_{int(time.time())}"
    job_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_EXEC),
        str(CROPPER_SCRIPT),
        "--weights", str(WEIGHTS_PATH),
        "--video", video_path,
        "--output-dir", str(job_output_dir),
        "--device", device,
        "--frame-interval", str(frame_interval),
        "--confidence", str(confidence),
        "--resize", str(resize),
        "--json-stream",
    ]
    parsed_cores = parse_cpu_cores(cpu_cores)
    env_extra = {}
    if parsed_cores is not None:
        cmd.extend(["--cpu-cores", str(parsed_cores)])
        env_extra["OMP_NUM_THREADS"] = str(parsed_cores)
        env_extra["TF_NUM_INTRAOP_THREADS"] = str(parsed_cores)
        env_extra["MKL_NUM_THREADS"] = str(parsed_cores)
        env_extra["OPENBLAS_NUM_THREADS"] = str(parsed_cores)

    if max_frames is not None and max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    if square:
        cmd.append("--square")
    if fast_mode is True:
        cmd.append("--fast-mode")
    elif fast_mode is False:
        cmd.append("--no-fast-mode")

    return StreamingResponse(stream_subprocess_output(cmd, env_extra=env_extra), media_type="text/event-stream")


@app.get("/api/stream-image")
async def stream_image(
    image_path: str = Query(...),
    device: str = Query("auto"),
    cpu_cores: Optional[str] = Query(None),
    confidence: float = Query(0.70),
    square: bool = Query(True),
    resize: int = Query(224),
    fast_mode: Optional[bool] = Query(None),
):
    """Processes an image file and streams results and logs over SSE."""
    if not os.path.exists(image_path):
        # Check test_samples
        sample_path = TEST_SAMPLES_DIR / image_path
        if sample_path.exists():
            image_path = str(sample_path)
        else:
            raise HTTPException(status_code=404, detail="Image file not found")

    job_output_dir = OUTPUT_DIR / f"run_img_{int(time.time())}"
    job_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_EXEC),
        str(CROPPER_SCRIPT),
        "--weights", str(WEIGHTS_PATH),
        "--input", image_path,
        "--output-dir", str(job_output_dir),
        "--device", device,
        "--confidence", str(confidence),
        "--resize", str(resize),
        "--json-stream",
    ]
    parsed_cores = parse_cpu_cores(cpu_cores)
    env_extra = {}
    if parsed_cores is not None:
        cmd.extend(["--cpu-cores", str(parsed_cores)])
        env_extra["OMP_NUM_THREADS"] = str(parsed_cores)
        env_extra["TF_NUM_INTRAOP_THREADS"] = str(parsed_cores)
        env_extra["MKL_NUM_THREADS"] = str(parsed_cores)
        env_extra["OPENBLAS_NUM_THREADS"] = str(parsed_cores)

    if square:
        cmd.append("--square")
    if fast_mode is True:
        cmd.append("--fast-mode")
    elif fast_mode is False:
        cmd.append("--no-fast-mode")

    return StreamingResponse(stream_subprocess_output(cmd, env_extra=env_extra), media_type="text/event-stream")


@app.get("/api/stream-benchmark")
async def stream_benchmark(max_images: int = Query(2)):
    """Runs comparative hardware benchmark and streams logs & results."""
    cmd = [
        str(PYTHON_EXEC),
        str(CROPPER_SCRIPT),
        "--weights", str(WEIGHTS_PATH),
        "--benchmark",
        "--max-images", str(max_images),
        "--json-stream",
    ]
    return StreamingResponse(stream_subprocess_output(cmd), media_type="text/event-stream")


def resolve_video_path(v_path: str) -> Path:
    p = Path(v_path)
    if p.exists():
        return p.resolve()
    cand_base = BASE_DIR / v_path
    if cand_base.exists():
        return cand_base.resolve()
    cand_root = Path("/home/adis_mayur/F_CODES") / v_path
    if cand_root.exists():
        return cand_root.resolve()
    cand_upload = UPLOADS_DIR / v_path
    if cand_upload.exists():
        return cand_upload.resolve()
    if not p.is_absolute():
        matches = sorted(UPLOADS_DIR.glob(f"*_{p.name}"))
        if matches:
            return matches[-1].resolve()
        matches_stem = sorted(UPLOADS_DIR.glob(f"*{p.stem}*.mp4"))
        if matches_stem:
            return matches_stem[-1].resolve()
    return p


@app.get("/api/stream-compare-videos")
async def stream_compare_videos(
    video_a: str = Query(...),
    video_b: str = Query(...),
    device: str = Query("gpu"),
    cpu_cores: Optional[str] = Query(None),
    upscale: str = Query("lanczos4"),
    scale: int = Query(2),
    threshold: int = Query(35),
    interval: int = Query(10),
    max_frames: int = Query(8),
    fast_mode: bool = Query(True),
):
    """Compares two cattle videos using Mask R-CNN crop + Lanczos-4 upscale + ORB matcher."""
    resolved_a = resolve_video_path(video_a)
    resolved_b = resolve_video_path(video_b)

    if not resolved_a.exists():
        raise HTTPException(status_code=404, detail=f"Video A file not found: {video_a}")
    if not resolved_b.exists():
        raise HTTPException(status_code=404, detail=f"Video B file not found: {video_b}")

    job_output_dir = OUTPUT_DIR / f"run_compare_{int(time.time()*1000)}"
    job_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_EXEC),
        str(MATCHER_SCRIPT),
        "--weights", str(WEIGHTS_PATH),
        "--video-a", str(resolved_a),
        "--video-b", str(resolved_b),
        "--output-dir", str(job_output_dir),
        "--device", device,
        "--upscale", upscale,
        "--scale", str(scale),
        "--threshold", str(threshold),
        "--interval", str(interval),
        "--max-frames", str(max_frames),
        "--json-stream",
    ]
    if fast_mode:
        cmd.append("--fast-mode")
    else:
        cmd.append("--no-fast-mode")

    parsed_cores = parse_cpu_cores(cpu_cores)
    env_extra = {}
    if parsed_cores is not None:
        cmd.extend(["--cpu-cores", str(parsed_cores)])
        env_extra["OMP_NUM_THREADS"] = str(parsed_cores)
        env_extra["TF_NUM_INTRAOP_THREADS"] = str(parsed_cores)
        env_extra["MKL_NUM_THREADS"] = str(parsed_cores)
        env_extra["OPENBLAS_NUM_THREADS"] = str(parsed_cores)

    return StreamingResponse(stream_subprocess_output(cmd, env_extra=env_extra), media_type="text/event-stream")


def resolve_image_path(img_path: str) -> Optional[Path]:
    """
    Resolves an image or mask path across absolute paths, output directory,
    test_samples, uploads, and workspace directories.
    """
    if not img_path or not str(img_path).strip():
        return None
    p = Path(img_path)
    if p.is_file():
        return p.resolve()

    clean_p = str(img_path).strip().lstrip("/")
    # Check /output/... or output/...
    if clean_p.startswith("output/"):
        rel = clean_p[len("output/"):]
        cand = OUTPUT_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check /test_samples/... or test_samples/...
    if clean_p.startswith("test_samples/"):
        rel = clean_p[len("test_samples/"):]
        cand = TEST_SAMPLES_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check /uploads/... or uploads/...
    if clean_p.startswith("uploads/"):
        rel = clean_p[len("uploads/"):]
        cand = UPLOADS_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check relative to OUTPUT_DIR
    cand = OUTPUT_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to BASE_DIR
    cand = BASE_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to TEST_SAMPLES_DIR
    cand = TEST_SAMPLES_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to UPLOADS_DIR
    cand = UPLOADS_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to /home/adis_mayur/F_CODES
    cand = Path("/home/adis_mayur/F_CODES") / clean_p
    if cand.is_file():
        return cand.resolve()

    return None


class GrooveTuneRequest(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    method: str = "blackhat"
    scales: List[int] = Field(default_factory=lambda: [9, 15, 21])
    threshold_method: str = "otsu"
    min_component_size: int = 15
    skeleton_enabled: bool = True


@app.post("/api/grooves/tune")
async def tune_grooves(req: GrooveTuneRequest):
    """
    Dedicated interactive tuning endpoint for cattle muzzle dark groove extraction & skeletonization.
    Allows real-time tuning of morphology method, scales, binarization, and noise pruning.
    """
    # 1. Parameter Validation
    method_clean = req.method.strip().lower()
    if method_clean not in ("blackhat", "frangi"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{req.method}'. Supported methods: 'blackhat', 'frangi'.",
        )

    thresh_clean = req.threshold_method.strip().lower()
    if thresh_clean not in ("otsu", "adaptive"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid threshold_method '{req.threshold_method}'. Supported methods: 'otsu', 'adaptive'.",
        )

    if not req.scales or not isinstance(req.scales, list):
        raise HTTPException(
            status_code=400,
            detail="Parameter 'scales' must be a non-empty list of integers (e.g. [9, 15, 21]).",
        )

    for s in req.scales:
        if not isinstance(s, int) or s < 1:
            raise HTTPException(
                status_code=400,
                detail=f"All scales must be positive integers >= 1, got {s}.",
            )

    if req.min_component_size < 0:
        raise HTTPException(
            status_code=400,
            detail=f"min_component_size must be >= 0, got {req.min_component_size}.",
        )

    # 2. Resolve image
    resolved_img_path = resolve_image_path(req.image_path)
    if not resolved_img_path or not resolved_img_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found at: '{req.image_path}'. Ensure file exists in output or test_samples.",
        )

    image_bgr = cv2.imread(str(resolved_img_path))
    if image_bgr is None or image_bgr.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image at '{req.image_path}'. File may be corrupt or not an image.",
        )

    # 3. Resolve mask (if provided)
    mask = None
    if req.mask_path:
        resolved_mask_path = resolve_image_path(req.mask_path)
        if not resolved_mask_path or not resolved_mask_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Mask file not found at: '{req.mask_path}'.",
            )
        mask = cv2.imread(str(resolved_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode mask file at '{req.mask_path}'.",
            )
        # Ensure dimensions match image
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(
                mask,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    # 4. Create unique output directory for this tune run
    tune_id = f"tune_{int(time.time() * 1000)}"
    tune_dir = OUTPUT_DIR / tune_id
    tune_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run extraction pipeline
        result = extract_muzzle_grooves_pipeline(
            image_bgr=image_bgr,
            mask=mask,
            method=method_clean,
            scales=req.scales,
            threshold_method=thresh_clean,
            min_component_size=req.min_component_size,
            exclude_nostrils=True,
            output_dir=None,
        )

        resp_img = result["enhanced_response"]
        groove_mask_img = result["groove_mask"]

        if req.skeleton_enabled:
            skel_img = result["groove_skeleton"]
            overlay_img = result["groove_overlay"]
            skel_pixel_count = int(result["num_skeleton_pixels"])
        else:
            skel_img = np.zeros_like(groove_mask_img)
            overlay_img = image_bgr.copy()
            skel_pixel_count = 0

        p_resp = tune_dir / "groove_response.png"
        p_mask = tune_dir / "groove_mask.png"
        p_skel = tune_dir / "groove_skeleton.png"
        p_over = tune_dir / "groove_overlay.jpg"

        cv2.imwrite(str(p_resp), resp_img)
        cv2.imwrite(str(p_mask), groove_mask_img)
        cv2.imwrite(str(p_skel), skel_img)
        cv2.imwrite(str(p_over), overlay_img)

        h_orig, w_orig = image_bgr.shape[:2]

        return {
            "success": True,
            "groove_response_url": f"/output/{tune_id}/groove_response.png",
            "groove_mask_url": f"/output/{tune_id}/groove_mask.png",
            "groove_skeleton_url": f"/output/{tune_id}/groove_skeleton.png",
            "groove_overlay_url": f"/output/{tune_id}/groove_overlay.jpg",
            "timings": {
                "total_ms": round(result["timing_ms"], 2),
            },
            "resolution": f"{w_orig}x{h_orig}",
            "groove_pixel_count": int(result["num_groove_pixels"]),
            "skeleton_pixel_count": skel_pixel_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Groove tuning pipeline execution failed: {str(e)}",
        )


@app.get("/api/grooves/sample-result")
async def get_sample_groove_result():
    """Returns pre-computed groove extraction for instant UI preview."""
    summary_path = OUTPUT_DIR / "grooves_muzzleVideo" / "00_groove_5stages_summary.jpg"
    native_path = OUTPUT_DIR / "grooves_muzzleVideo" / "01_native_muzzle_crop.jpg"
    if summary_path.exists() and native_path.exists():
        return {
            "success": True,
            "video_name": "muzzleVideo.mp4",
            "frame_idx": 0,
            "confidence": 0.998,
            "quality": {"composite": 61.3, "sharpness": 172.5},
            "resolution": "392x392",
            "method": "blackhat",
            "scales": [9, 15, 21],
            "threshold_method": "otsu",
            "min_component_size": 15,
            "groove_pixel_count": 9497,
            "skeleton_pixel_count": 2815,
            "groove_coverage_pct": 22.36,
            "crop_path": str(native_path),
            "artifacts": {
                "summary_panel": "/output/grooves_muzzleVideo/00_groove_5stages_summary.jpg",
                "native_crop": "/output/grooves_muzzleVideo/01_native_muzzle_crop.jpg",
                "enhanced_ridge": "/output/grooves_muzzleVideo/02_lanczos_clahe_ridge.jpg",
                "groove_response": "/output/grooves_muzzleVideo/groove_response.png",
                "groove_mask": "/output/grooves_muzzleVideo/groove_mask.png",
                "groove_skeleton": "/output/grooves_muzzleVideo/groove_skeleton.png",
                "groove_overlay": "/output/grooves_muzzleVideo/groove_overlay.png",
            },
        }
    raise HTTPException(status_code=404, detail="No precomputed sample result found.")


class ProcessVideoGroovesRequest(BaseModel):
    video_path: str
    method: str = "blackhat"
    scales: List[int] = Field(default_factory=lambda: [9, 15, 21])
    threshold_method: str = "otsu"
    min_component_size: int = 15


@app.post("/api/grooves/process-video")
async def process_video_grooves_endpoint(req: ProcessVideoGroovesRequest):
    """Processes a cattle video end-to-end to extract muzzle grooves and skeletons."""
    resolved_v = resolve_video_path(req.video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    import asyncio
    from functools import partial
    from process_video_grooves import process_video_grooves

    loop = asyncio.get_event_loop()
    fn = partial(
        process_video_grooves,
        video_path=str(resolved_v),
        method=req.method,
        scales=req.scales,
        threshold_method=req.threshold_method,
        min_component_size=req.min_component_size,
    )
    try:
        res = await loop.run_in_executor(None, fn)
        stem = resolved_v.stem
        rel_dir = f"grooves_{stem}"
        res["artifacts"] = {
            k: f"/output/{rel_dir}/{Path(v).name}"
            for k, v in res["artifacts"].items()
        }
        res["crop_path"] = str(OUTPUT_DIR / rel_dir / "01_native_muzzle_crop.jpg")
        res["success"] = True
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


# =====================================================================
# MULTI-IMAGE BIOMETRIC REFERENCE TEMPLATE ENDPOINTS
# =====================================================================

class BuildTemplateBatchRequest(BaseModel):
    image_paths: List[str]
    template_id: Optional[str] = None
    duplicate_threshold: float = 0.92
    method: str = "blackhat"
    scales: List[int] = Field(default_factory=lambda: [9, 15, 21])
    threshold_method: str = "otsu"
    min_component_size: int = 15


@app.post("/api/template/build-from-batch")
async def build_template_from_batch(req: BuildTemplateBatchRequest):
    """
    Ingests up to 5 images, assesses quality, extracts topological groove graphs,
    and consolidates them into a robust reference template.
    """
    import asyncio
    from functools import partial

    if not req.image_paths:
        raise HTTPException(status_code=400, detail="At least 1 image path required.")

    # Resolve paths
    resolved_paths = []
    for p_str in req.image_paths[:5]:
        p = Path(p_str)
        if not p.is_absolute():
            p = BASE_DIR / p_str
        if not p.exists():
            cand1 = TEST_SAMPLES_DIR / Path(p_str).name
            cand2 = OUTPUT_DIR / p_str
            if cand1.exists():
                p = cand1
            elif cand2.exists():
                p = cand2
            else:
                raise HTTPException(status_code=404, detail=f"Image not found: {p_str}")
        resolved_paths.append(str(p))

    tmpl_id = req.template_id or f"tmpl_{int(time.time() * 1000)}"
    batch_out_dir = TEMPLATES_DIR / tmpl_id
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    def _execute_batch():
        batch_res = process_image_batch(
            images=resolved_paths,
            output_dir=batch_out_dir,
            groove_params={
                "method": req.method,
                "scales": req.scales,
                "threshold_method": req.threshold_method,
                "min_component_size": req.min_component_size,
            },
            duplicate_threshold=req.duplicate_threshold,
            batch_id=tmpl_id,
        )

        records = batch_res["records"]
        for idx, r in enumerate(records):
            if r.get("is_usable") and r.get("groove_skeleton") is not None:
                g_res = extract_skeleton_graph(r["groove_skeleton"], mask=r.get("clean_mask"))
                r["descriptors"] = g_res["descriptors"]
                r["branches"] = g_res["branches"]
                r["centroid"] = g_res["spatial_centroid"]

                p_overlay = batch_out_dir / f"img_{idx:02d}_graph_overlay.png"
                overlay_img = visualize_groove_graph(r["masked_image"], g_res)
                cv2.imwrite(str(p_overlay), overlay_img)
                if "saved_paths" not in r:
                    r["saved_paths"] = {}
                r["saved_paths"]["graph_overlay"] = str(p_overlay)
            else:
                r["descriptors"] = {}
                r["branches"] = []

        template = build_consensus_template(records, template_id=tmpl_id)
        save_reference_template(template, batch_out_dir / f"{tmpl_id}.json")
        export_all_template_reports(template, batch_out_dir)

        rel_base = f"/output/templates/{tmpl_id}"
        template["artifacts"] = {
            "template_json_url": f"{rel_base}/{tmpl_id}.json",
            "measurements_csv_url": f"{rel_base}/per_image_measurements.csv",
            "consensus_csv_url": f"{rel_base}/consensus_summary.csv",
        }
        for idx, s in enumerate(template.get("source_images", [])):
            s["image_url"] = f"{rel_base}/img_{idx:02d}_native_crop.png"
            s["skeleton_url"] = f"{rel_base}/img_{idx:02d}_groove_skeleton.png"
            s["graph_overlay_url"] = f"{rel_base}/img_{idx:02d}_graph_overlay.png"

        return template

    loop = asyncio.get_event_loop()
    try:
        template = await loop.run_in_executor(None, _execute_batch)
        return template
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch template building failed: {str(e)}")


@app.get("/api/template/sample-batch")
async def get_sample_template():
    """Returns pre-computed or generated 5-image reference template for instant UI display."""
    sample_json = TEMPLATES_DIR / "sample_5image_consensus" / "sample_5image_consensus.json"
    if sample_json.exists():
        try:
            return load_reference_template(sample_json)
        except Exception:
            pass

    sample_images = list(TEST_SAMPLES_DIR.glob("*.jpg"))
    if not sample_images:
        sample_images = list(OUTPUT_DIR.glob("**/*crop*.jpg"))

    if not sample_images:
        return make_empty_template(template_id="sample_template_default", submitted_count=5)

    img_paths = [str(p) for p in (sample_images * 5)[:5]]
    req = BuildTemplateBatchRequest(image_paths=img_paths, template_id="sample_5image_consensus")
    return await build_template_from_batch(req)


@app.post("/api/template/save")
async def save_template_endpoint(data: Dict[str, Any] = Body(...)):
    """Saves reference template JSON to storage."""
    try:
        t = data.get("template") or data
        t_id = t.get("template_id") or f"tmpl_{int(time.time() * 1000)}"
        out_p = TEMPLATES_DIR / f"{t_id}.json"
        path_str = save_reference_template(t, out_p)
        return {"success": True, "template_id": t_id, "file_path": path_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save template: {str(e)}")


@app.get("/api/template/list")
async def list_templates_endpoint():
    """Lists saved reference templates."""
    try:
        templates = list_saved_templates(TEMPLATES_DIR)
        return {"success": True, "templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


class CompareTemplatesRequest(BaseModel):
    template_a: Any
    template_b: Any
    decision_threshold: Optional[float] = 0.72


@app.post("/api/template/compare")
async def compare_templates_endpoint(req: CompareTemplatesRequest):
    """Compares two reference templates and returns statistical biometric match verdict."""
    try:
        if isinstance(req.template_a, str):
            t_a = load_reference_template(req.template_a, storage_dir=TEMPLATES_DIR)
        else:
            t_a = req.template_a

        if isinstance(req.template_b, str):
            t_b = load_reference_template(req.template_b, storage_dir=TEMPLATES_DIR)
        else:
            t_b = req.template_b

        res = compare_two_templates(
            t_a,
            t_b,
            decision_threshold=req.decision_threshold or 0.72,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Template comparison failed: {str(e)}")


@app.get("/api/template/export/json/{template_id}")
async def export_template_json_endpoint(template_id: str):
    """Downloads canonical JSON reference template."""
    p = TEMPLATES_DIR / f"{template_id}.json"
    if not p.exists():
        p = TEMPLATES_DIR / template_id / f"{template_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return FileResponse(p, media_type="application/json", filename=f"{template_id}.json")


@app.get("/api/template/export/csv/{export_type}/{template_id}")
async def export_template_csv_endpoint(export_type: str, template_id: str):
    """Downloads per-image measurements or consensus summary CSV."""
    dir_p = TEMPLATES_DIR / template_id
    if export_type == "measurements":
        fname = "per_image_measurements.csv"
    elif export_type == "consensus":
        fname = "consensus_summary.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid export_type. Choose 'measurements' or 'consensus'")

    file_p = dir_p / fname
    if not file_p.exists():
        raise HTTPException(status_code=404, detail=f"CSV file {fname} not found for template {template_id}")
    return FileResponse(file_p, media_type="text/csv", filename=f"{template_id}_{fname}")


# =====================================================================
# 2-VIDEO 5-FRAME GROOVE PARAMETERS & COSINE SIMILARITY ENDPOINTS
# =====================================================================

# Cache holds NumPy arrays only in backend memory. JSON responses are always plain data.
from collections import OrderedDict
import hashlib
from functools import partial

_EXTRACTION_LOCK = asyncio.Lock()
_EXTRACTION_CACHE = OrderedDict()
_EXTRACTION_CACHE_MAX = 8

def _json_safe_frame(frame):
    return {k: v for k, v in frame.items() if not isinstance(v, np.ndarray)}


def _spatial_cache_key(video_path, method, threshold_method, adaptive_block_size, adaptive_c,
                       exclude_nostrils, min_spur_length, min_component_size, device, cpu_cores,
                       dl_backbone="convnext_tiny", dl_input_mode="composite_3ch"):
    stat = Path(video_path).stat()
    signature = f"{Path(video_path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{method}:{threshold_method}:{adaptive_block_size}:{adaptive_c}:{exclude_nostrils}:{min_spur_length}:{min_component_size}:{device}:{cpu_cores}:{dl_backbone}:{dl_input_mode}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _put_extraction_cache(key, frames):
    _EXTRACTION_CACHE[key] = frames
    _EXTRACTION_CACHE.move_to_end(key)
    while len(_EXTRACTION_CACHE) > _EXTRACTION_CACHE_MAX:
        _EXTRACTION_CACHE.popitem(last=False)


def _get_extraction_cache(key):
    frames = _EXTRACTION_CACHE.get(key)
    if frames is not None:
        _EXTRACTION_CACHE.move_to_end(key)
    return frames


def _extract_video_for_spatial_match(video_path, req):
    """Synchronous extraction helper; caller controls the extraction lock."""
    return extract_five_frames_groove_features(
        video_path=str(video_path),
        method=req.filter_method or "frangi",
        threshold_method=req.threshold_method or "adaptive",
        adaptive_block_size=int(req.adaptive_block_size or 15),
        adaptive_c=int(req.adaptive_c or 2),
        exclude_nostrils=bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
        device=req.device or "cpu",
        cpu_cores=int(req.cpu_cores or 6),
        min_spur_length=int(req.min_spur_length or 8),
        min_component_size=int(req.min_component_size or 30),
        dl_backbone=getattr(req, "dl_backbone", "convnext_tiny") or "convnext_tiny",
        dl_input_mode=getattr(req, "dl_input_mode", "composite_3ch") or "composite_3ch",
        enable_dl_embedding=True,
    )

class ExtractVideoFiveFramesRequest(BaseModel):
    video_path: str
    method: Optional[str] = "frangi"                     # 'frangi' (Frangi Hessian) or 'blackhat'
    threshold_method: Optional[str] = "adaptive"         # 'adaptive' (Adaptive Gaussian) or 'otsu' (Standard)
    adaptive_block_size: Optional[int] = 15
    adaptive_c: Optional[int] = 2
    exclude_nostrils: Optional[bool] = True
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6
    min_spur_length: Optional[int] = 8
    min_component_size: Optional[int] = 30
    dl_backbone: Optional[str] = "convnext_tiny"         # 'convnext_tiny' or 'resnet50'
    dl_input_mode: Optional[str] = "composite_3ch"       # 'composite_3ch', 'skeleton_only', 'groove_map'
    enable_dl_embedding: Optional[bool] = True

@app.post("/api/video/extract-five-muzzle-parameters")
async def extract_video_five_frames_endpoint(req: ExtractVideoFiveFramesRequest):
    """
    Extracts 5 muzzle images, ten topological parameters, and 512-D Deep Learning Biometric Embeddings.
    The backend caches internal NumPy arrays for spatial matching and strips those arrays from the HTTP response.
    """
    resolved_v = resolve_video_path(req.video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    from functools import partial
    fn = partial(
        extract_five_frames_groove_features,
        video_path=str(resolved_v),
        method=req.method or "frangi",
        threshold_method=req.threshold_method or "adaptive",
        adaptive_block_size=int(req.adaptive_block_size or 15),
        adaptive_c=int(req.adaptive_c or 2),
        exclude_nostrils=bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
        device=req.device or "cpu",
        cpu_cores=int(req.cpu_cores or 6),
        min_spur_length=int(req.min_spur_length or 8),
        min_component_size=int(req.min_component_size or 30),
        dl_backbone=req.dl_backbone or "convnext_tiny",
        dl_input_mode=req.dl_input_mode or "composite_3ch",
        enable_dl_embedding=bool(req.enable_dl_embedding if req.enable_dl_embedding is not None else True),
    )
    try:
        async with _EXTRACTION_LOCK:
            res = await asyncio.to_thread(fn)
        cache_key = _spatial_cache_key(
            resolved_v,
            req.method or "frangi",
            req.threshold_method or "adaptive",
            int(req.adaptive_block_size or 15),
            int(req.adaptive_c or 2),
            bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
            int(req.min_spur_length or 8),
            int(req.min_component_size or 30),
            req.device or "cpu",
            int(req.cpu_cores or 6),
            req.dl_backbone or "convnext_tiny",
            req.dl_input_mode or "composite_3ch",
        )
        _put_extraction_cache(cache_key, res.get("frames") or [])
        safe = dict(res)
        safe["frames"] = [_json_safe_frame(frame) for frame in (res.get("frames") or [])]
        return safe
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"5-Frame extraction failed: {exc}")


class CosineSimilarityRequest(BaseModel):
    vector_a: List[float]
    vector_b: List[float]
    dl_emb_a: Optional[Union[List[float], Dict[str, Any]]] = None
    dl_emb_b: Optional[Union[List[float], Dict[str, Any]]] = None
    dl_mean_a: Optional[List[float]] = None
    dl_mean_b: Optional[List[float]] = None
    dl_median_a: Optional[List[float]] = None
    dl_median_b: Optional[List[float]] = None
    per_frames_a: Optional[List[List[float]]] = None
    per_frames_b: Optional[List[List[float]]] = None
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6


@app.post("/api/video/cosine-similarity")
@app.post("/api/video/dl-cosine-similarity")
async def compute_cosine_similarity_endpoint(req: CosineSimilarityRequest):
    """
    Computes Normalized Cosine Similarity on classical 10-D graph minutiae parameters
    (with Bray-Curtis magnitude penalty calibration) and Deep Learning Biometric
    Embeddings (ConvNeXt-Tiny / ResNet-50 Spatial GeM & ArcFace).
    Supports dual embeddings (mean and median pooling) with full backward compatibility.
    """
    t0 = time.perf_counter()
    try:
        res = compute_cosine_similarity(
            vec_a=req.vector_a,
            vec_b=req.vector_b,
            dl_emb_a=req.dl_emb_a,
            dl_emb_b=req.dl_emb_b,
            per_frames_a=req.per_frames_a,
            per_frames_b=req.per_frames_b,
            dl_mean_a=req.dl_mean_a,
            dl_mean_b=req.dl_mean_b,
            dl_median_a=req.dl_median_a,
            dl_median_b=req.dl_median_b,
        )
        t_sim_ms = (time.perf_counter() - t0) * 1000
        if "step_timings" not in res or not isinstance(res["step_timings"], dict):
            res["step_timings"] = {}
        res["step_timings"]["step_7_consensus_ms"] = max(0.5, round(t_sim_ms, 1))
        res["step_timings"]["t_consensus_ms"] = res.get("t_consensus_ms", max(0.5, round(t_sim_ms, 1)))
        res["step_timings"]["total_ms"] = max(0.5, round(t_sim_ms, 1))
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cosine similarity calculation failed: {exc}")


def _extract_embedding_from_summary(summary_data: Dict[str, Any], pool_method: str = "median") -> Optional[List[float]]:
    """Helper to extract an L2-normalized 512-D embedding list from summary data."""
    if not isinstance(summary_data, dict):
        return None
    dl = summary_data.get("dl_embedding")
    if not dl or not isinstance(dl, dict):
        return None

    emb = None
    if pool_method == "mean":
        emb = dl.get("mean_embedding") or dl.get("median_embedding")
    else:
        emb = dl.get("median_embedding") or dl.get("mean_embedding")

    if not emb and "per_frame_embeddings" in dl and dl["per_frame_embeddings"]:
        frames = dl["per_frame_embeddings"]
        if pool_method == "mean":
            emb = np.mean(frames, axis=0).tolist()
        else:
            emb = np.median(frames, axis=0).tolist()

    if emb:
        arr = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        return [round(float(x), 5) for x in arr]
    return None


GALLERY_PRESET_DEFINITIONS = [
    {
        "id": "cattle_001_a",
        "name": "Cattle A (Profile 1)",
        "folders": ["muzzleVideo_frangi_adaptive", "1793364335_muzzleVideo_frangi_adaptive", "muzzleVideo"],
        "video_path": "uploads/1793364335_muzzleVideo.mp4",
        "notes": "Registered Baseline",
        "tag": "US-94210-A",
        "breed": "Holstein Friesian",
        "registered_date": "2024-09-10",
        "meta": {"source_video": "muzzleVideo.mp4", "breed": "Holstein Friesian", "registration_id": "RF-9021", "animal_id": "cattle_001"},
    },
    {
        "id": "cattle_001_b",
        "name": "Cattle A (Profile 2 - Same Animal)",
        "folders": ["muzzleVideo2_frangi_adaptive", "1793364192_muzzleVideo2_frangi_adaptive", "muzzleVideo2"],
        "video_path": "uploads/1793364192_muzzleVideo2.mp4",
        "notes": "Alternative Take",
        "tag": "US-94210-B",
        "breed": "Holstein Friesian",
        "registered_date": "2024-09-12",
        "meta": {"source_video": "muzzleVideo2.mp4", "breed": "Holstein Friesian", "registration_id": "RF-9021", "animal_id": "cattle_001"},
    },
    {
        "id": "cattle_002",
        "name": "Cattle B (Different Animal)",
        "folders": ["muzzleVideo3_frangi_adaptive", "1793364199_muzzleVideo3_frangi_adaptive", "muzzleVideo3"],
        "video_path": "uploads/1793364199_muzzleVideo3.mp4",
        "notes": "Unrelated Dairy Cow",
        "tag": "US-88412-C",
        "breed": "Jersey",
        "registered_date": "2024-09-15",
        "meta": {"source_video": "muzzleVideo3.mp4", "breed": "Jersey", "registration_id": "RF-4412", "animal_id": "cattle_002"},
    },
    {
        "id": "cattle_003",
        "name": "Cattle C (Heifer)",
        "folders": ["1793369778_muzzleVideo4_frangi_adaptive", "muzzleVideo4_frangi_adaptive"],
        "video_path": "uploads/1793369778_muzzleVideo4.mp4",
        "notes": "Registered Heifer",
        "tag": "US-77201-D",
        "breed": "Angus",
        "registered_date": "2024-09-20",
        "meta": {"source_video": "muzzleVideo4.mp4", "breed": "Angus", "registration_id": "RF-7720", "animal_id": "cattle_003"},
    },
    {
        "id": "cattle_004",
        "name": "Cattle D (Calf)",
        "folders": ["1793369565_muzzleVideo5_frangi_adaptive", "1793369773_muzzleVideo5_frangi_adaptive", "muzzleVideo5_frangi_adaptive"],
        "video_path": "uploads/1793369565_muzzleVideo5.mp4",
        "notes": "Pasture Herd #4",
        "tag": "US-31085-E",
        "breed": "Guernsey",
        "registered_date": "2024-10-01",
        "meta": {"source_video": "muzzleVideo5.mp4", "breed": "Guernsey", "registration_id": "RF-3108", "animal_id": "cattle_004"},
    },
    {
        "id": "cattle_005",
        "name": "Cattle E (Crossbreed)",
        "folders": ["1793364507_muzzleVideo7_frangi_adaptive", "muzzleVideo7_frangi_adaptive"],
        "video_path": "uploads/1793364507_muzzleVideo7.mp4",
        "notes": "Breeding Stock",
        "tag": "US-88540-F",
        "breed": "Simmental Cross",
        "registered_date": "2024-10-05",
        "meta": {"source_video": "muzzleVideo7.mp4", "breed": "Simmental Cross", "registration_id": "RF-8854", "animal_id": "cattle_005"},
    },
]

_CACHED_GALLERY_PRESETS: Optional[List[Dict[str, Any]]] = None

def load_gallery_presets_data(force_reload: bool = False) -> List[Dict[str, Any]]:
    global _CACHED_GALLERY_PRESETS
    if _CACHED_GALLERY_PRESETS is not None and not force_reload:
        return _CACHED_GALLERY_PRESETS

    presets_json_path = BASE_DIR / "config" / "gallery_presets.json"
    if presets_json_path.exists():
        try:
            preset_content = json.loads(presets_json_path.read_text(encoding="utf-8"))
            gallery = preset_content.get("gallery", [])
            if gallery and len(gallery) > 0:
                _CACHED_GALLERY_PRESETS = gallery
                return gallery
        except Exception as exc:
            logger.warning(f"Failed loading presets JSON from {presets_json_path}: {exc}")

    vff_dir = OUTPUT_DIR / "video_five_frames"
    gallery = []

    for pdef in GALLERY_PRESET_DEFINITIONS:
        cid = pdef["id"]
        name = pdef["name"]
        notes = pdef.get("notes")
        meta = dict(pdef.get("meta") or {})
        emb = None

        # 1. Search preset folders
        for folder_name in pdef.get("folders", []):
            summary_path = vff_dir / folder_name / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    emb = _extract_embedding_from_summary(data)
                    if emb:
                        meta["source_summary"] = str(summary_path.relative_to(BASE_DIR))
                        break
                except Exception as exc:
                    logger.debug(f"Could not load {summary_path}: {exc}")

        # 2. Search any summary in vff_dir matching the stem
        if emb is None:
            stem = pdef.get("video_path", "").split("/")[-1].replace(".mp4", "")
            if stem:
                for cand_summary in sorted(vff_dir.glob(f"*{stem}*/summary.json")):
                    try:
                        data = json.loads(cand_summary.read_text(encoding="utf-8"))
                        emb = _extract_embedding_from_summary(data)
                        if emb:
                            meta["source_summary"] = str(cand_summary.relative_to(BASE_DIR))
                            break
                    except Exception:
                        pass

        if emb:
            gallery.append({
                "id": cid,
                "name": name,
                "embedding": emb,
                "notes": notes,
                "tag": pdef.get("tag"),
                "breed": pdef.get("breed"),
                "registered_date": pdef.get("registered_date"),
                "meta": meta,
            })

    _CACHED_GALLERY_PRESETS = gallery
    return gallery


@app.get("/api/embedding/gallery-presets")
async def get_embedding_gallery_presets_endpoint(reload: bool = Query(False)):
    """
    Returns a pre-packaged gallery of registered cattle embeddings from
    existing summary files in output/video_five_frames/ (e.g. muzzleVideo_frangi_adaptive
    as Cattle A Profile 1, muzzleVideo2_frangi_adaptive as Cattle A Profile 2,
    muzzleVideo3_frangi_adaptive as Cattle B) or compute them on demand.
    """
    gallery = load_gallery_presets_data(force_reload=reload)

    # Check if samples defined in config/gallery_presets.json
    presets_json_path = BASE_DIR / "config" / "gallery_presets.json"
    sample_a = None
    sample_b = None
    if presets_json_path.exists():
        try:
            raw_data = json.loads(presets_json_path.read_text(encoding="utf-8"))
            sample_a = raw_data.get("sample_cow_a")
            sample_b = raw_data.get("sample_cow_b")
        except Exception:
            pass

    if not sample_a and gallery:
        for cand in gallery:
            if "cattle_001_a" in cand["id"] or "profile 1" in cand["name"].lower():
                sample_a = {"id": cand["id"], "name": cand["name"], "embedding": cand["embedding"]}
                break
        if not sample_a:
            sample_a = {"id": gallery[0]["id"], "name": gallery[0]["name"], "embedding": gallery[0]["embedding"]}

    if not sample_b and len(gallery) > 1:
        for cand in gallery:
            if "cattle_002" in cand["id"] or "cattle b" in cand["name"].lower():
                sample_b = {"id": cand["id"], "name": cand["name"], "embedding": cand["embedding"]}
                break
        if not sample_b:
            sample_b = {"id": gallery[1]["id"], "name": gallery[1]["name"], "embedding": gallery[1]["embedding"]}

    return {
        "success": True,
        "gallery": gallery,
        "sample_cow_a": sample_a,
        "sample_cow_b": sample_b,
    }


class OneToNMatchRequest(BaseModel):
    query_embedding: List[float]
    gallery: Optional[List[Dict[str, Any]]] = None
    similarity_metric: Optional[str] = "cosine"  # "cosine" or "l2"
    threshold: Optional[float] = 88.0
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6


@app.post("/api/embedding/1-to-n-match")
async def one_to_n_match_endpoint(req: OneToNMatchRequest):
    """
    1-to-N Multi-Cattle Biometric Matching Endpoint:
    Matches a query 512-D embedding against a gallery matrix G (N, 512).
    Vectorized NumPy dot-product computation, Euclidean L2 distance,
    biometric thresholding verdicts, runner-up margin, and candidate ranking.
    """
    t_match_0 = time.perf_counter()
    if not req.query_embedding or len(req.query_embedding) == 0:
        raise HTTPException(status_code=400, detail="query_embedding cannot be empty.")

    # 1. Normalize query vector to unit length: q = q / max(1e-8, ||q||_2)
    q = np.array(req.query_embedding, dtype=np.float32)
    dim = int(len(q))
    q_norm = float(np.linalg.norm(q))
    q = q / max(1e-8, q_norm)

    gallery_candidates = req.gallery
    if gallery_candidates is None:
        gallery_candidates = load_gallery_presets_data()

    N = len(gallery_candidates)
    if N == 0:
        return {
            "success": True,
            "query_dim": dim,
            "total_candidates": 0,
            "top_match": None,
            "margin_to_runner_up": None,
            "ranked_results": [],
            "margin_ahead_pct": 0.0,
            "leaderboard": [],
            "total_gallery_candidates": 0,
            "verdict": "NO MATCH (Empty Gallery)",
            "verdict_is_match": False,
            "verdict_color": "rose",
        }

    # 2. Extract and validate candidate embeddings
    candidate_embeddings = []
    for idx, item in enumerate(gallery_candidates):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Gallery candidate at index {idx} must be a dictionary.")
        raw_emb = item.get("embedding")
        if raw_emb is None:
            raise HTTPException(status_code=400, detail=f"Candidate at index {idx} missing 'embedding' field.")
        if isinstance(raw_emb, dict):
            raw_emb = raw_emb.get("median_embedding") or raw_emb.get("mean_embedding")
            if not raw_emb:
                raise HTTPException(status_code=400, detail=f"Candidate at index {idx} embedding object has no median/mean vector.")
        if len(raw_emb) != dim:
            raise HTTPException(
                status_code=400,
                detail=f"Dimension mismatch at candidate {idx} ('{item.get('id', idx)}'): query has {dim} dimensions, candidate has {len(raw_emb)}.",
            )
        candidate_embeddings.append(raw_emb)

    # 3. Stack gallery embeddings into matrix G (N, 512), normalize each row to unit length
    G = np.array(candidate_embeddings, dtype=np.float32)
    row_norms = np.linalg.norm(G, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    G = G / row_norms

    # 4. Compute dot products in one vectorized NumPy call
    sims = np.dot(G, q)  # (N,)
    sim_pcts = np.clip(sims, 0.0, 1.0) * 100.0
    l2_dists = np.linalg.norm(G - q, axis=1)  # (N,)

    # 5. For each candidate, determine verdict
    threshold = float(req.threshold if req.threshold is not None else 88.0)
    ranked_results = []
    for i, item in enumerate(gallery_candidates):
        s_pct = float(sim_pcts[i])
        s_val = float(sims[i])
        dist = float(l2_dists[i])

        if s_pct >= 88.0:
            verdict = "MATCH (Confirmed Same Cattle)"
            badge = "match"
            color = "emerald"
        elif s_pct >= 78.0:
            verdict = "LIKELY MATCH"
            badge = "likely"
            color = "teal"
        elif s_pct >= 65.0:
            verdict = "INCONCLUSIVE"
            badge = "inconclusive"
            color = "amber"
        else:
            verdict = "DISTINCT (Different Animals)"
            badge = "distinct"
            color = "rose"

        meta_dict = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        cand_dict = {
            "id": str(item.get("id", f"candidate_{i}")),
            "name": str(item.get("name", f"Candidate {i+1}")),
            "similarity": round(s_val, 4),
            "similarity_pct": round(s_pct, 2),
            "l2_distance": round(dist, 4),
            "verdict": verdict,
            "badge": badge,
            "color": color,
            "verdict_color": color,
            "notes": item.get("notes"),
            "tag": item.get("tag") or meta_dict.get("tag") or "N/A",
            "breed": item.get("breed") or meta_dict.get("breed") or "Unknown",
            "registered_date": item.get("registered_date") or meta_dict.get("registered_date") or "",
            "meta": meta_dict,
            "is_match": bool(s_pct >= threshold),
        }
        ranked_results.append(cand_dict)

    # 6. Rank candidates descending by similarity score
    ranked_results.sort(key=lambda x: x["similarity_pct"], reverse=True)
    for rank_idx, r in enumerate(ranked_results, start=1):
        r["rank"] = rank_idx

    top_match = ranked_results[0] if ranked_results else None
    margin_to_runner_up = (
        round(float(ranked_results[0]["similarity_pct"] - ranked_results[1]["similarity_pct"]), 2)
        if len(ranked_results) > 1
        else None
    )

    is_confirmed = bool(top_match and top_match["similarity_pct"] >= threshold)
    return {
        "success": True,
        "query_dim": dim,
        "total_candidates": len(ranked_results),
        "top_match": top_match,
        "margin_to_runner_up": margin_to_runner_up,
        "ranked_results": ranked_results,
        # Backward compatibility and UI integration fields
        "margin_ahead_pct": margin_to_runner_up if margin_to_runner_up is not None else 0.0,
        "leaderboard": ranked_results,
        "total_gallery_candidates": len(ranked_results),
        "verdict": f"MATCH FOUND: {top_match['name']}" if is_confirmed else "NO MATCH (All Distinct Animals)",
        "verdict_is_match": is_confirmed,
        "verdict_color": "emerald" if is_confirmed else "rose",
        "threshold": threshold,
        "step_timings": {
            "step_7_consensus_ms": max(0.5, round((time.perf_counter() - t_match_0) * 1000, 1)),
            "total_ms": max(0.5, round((time.perf_counter() - t_match_0) * 1000, 1)),
        },
    }


# Backwards compatibility aliases
match_one_to_n_endpoint = one_to_n_match_endpoint
get_gallery_presets_endpoint = get_embedding_gallery_presets_endpoint


class SingleVideoEmbeddingExportRequest(BaseModel):
    video_path: str
    pool_method: Optional[str] = "median"  # 'median' or 'mean'
    method: Optional[str] = "frangi"
    threshold_method: Optional[str] = "adaptive"
    adaptive_block_size: Optional[int] = 15
    adaptive_c: Optional[int] = 2
    exclude_nostrils: Optional[bool] = True
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6
    min_spur_length: Optional[int] = 8
    min_component_size: Optional[int] = 30
    dl_backbone: Optional[str] = "convnext_tiny"
    dl_input_mode: Optional[str] = "composite_3ch"


async def _handle_single_video_embedding_export(
    video_path: str,
    pool_method: str = "median",
    req_opts: Optional[SingleVideoEmbeddingExportRequest] = None,
) -> Dict[str, Any]:
    resolved_v = resolve_video_path(video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    chosen_pool = pool_method or "median"

    # Search existing summary.json in output/video_five_frames/
    vff_dir = OUTPUT_DIR / "video_five_frames"
    stem = resolved_v.stem
    cand_summaries = list(vff_dir.glob(f"*{stem}*/summary.json"))
    name_no_ts = "_".join(stem.split("_")[1:]) if "_" in stem else stem
    if name_no_ts and name_no_ts != stem:
        cand_summaries.extend(vff_dir.glob(f"*{name_no_ts}*/summary.json"))

    for c_path in sorted(cand_summaries, reverse=True):
        try:
            data = json.loads(c_path.read_text(encoding="utf-8"))
            dl = data.get("dl_embedding")
            emb = _extract_embedding_from_summary(data, pool_method=chosen_pool)
            if emb:
                return {
                    "success": True,
                    "video_name": resolved_v.name,
                    "video_path": str(resolved_v),
                    "embedding": emb,
                    "embedding_dim": len(emb),
                    "pool_method": chosen_pool,
                    "intra_video_consistency_pct": dl.get("intra_video_consistency_pct") if dl else None,
                    "cached": True,
                    "summary_path": str(c_path.relative_to(BASE_DIR)),
                    "dl_summary": {
                        "backbone": dl.get("backbone") if dl else "convnext_tiny",
                        "input_mode": dl.get("input_mode") if dl else "composite_3ch",
                        "weights_mode": dl.get("weights_mode") if dl else "pretrained_gem_zeroshot",
                        "avg_inference_ms": dl.get("avg_inference_ms") if dl else None,
                    } if dl else {},
                }
        except Exception as exc:
            logger.debug(f"Error reading summary {c_path}: {exc}")

    # Not cached: compute on demand
    from functools import partial
    fn = partial(
        extract_five_frames_groove_features,
        video_path=str(resolved_v),
        method=req_opts.method if req_opts else "frangi",
        threshold_method=req_opts.threshold_method if req_opts else "adaptive",
        adaptive_block_size=int(req_opts.adaptive_block_size or 15) if req_opts else 15,
        adaptive_c=int(req_opts.adaptive_c or 2) if req_opts else 2,
        exclude_nostrils=bool(req_opts.exclude_nostrils if (req_opts and req_opts.exclude_nostrils is not None) else True),
        device=req_opts.device if req_opts else "cpu",
        cpu_cores=int(req_opts.cpu_cores or 6) if req_opts else 6,
        min_spur_length=int(req_opts.min_spur_length or 8) if req_opts else 8,
        min_component_size=int(req_opts.min_component_size or 30) if req_opts else 30,
        dl_backbone=req_opts.dl_backbone if req_opts else "convnext_tiny",
        dl_input_mode=req_opts.dl_input_mode if req_opts else "composite_3ch",
        enable_dl_embedding=True,
    )
    async with _EXTRACTION_LOCK:
        res = await asyncio.to_thread(fn)

    emb = _extract_embedding_from_summary(res, pool_method=chosen_pool)
    if not emb:
        raise HTTPException(status_code=500, detail="Failed to extract biometric embedding from video.")

    dl = res.get("dl_embedding") or {}
    return {
        "success": True,
        "video_name": resolved_v.name,
        "video_path": str(resolved_v),
        "embedding": emb,
        "embedding_dim": len(emb),
        "pool_method": chosen_pool,
        "intra_video_consistency_pct": dl.get("intra_video_consistency_pct"),
        "cached": False,
        "dl_summary": {
            "backbone": dl.get("backbone"),
            "input_mode": dl.get("input_mode"),
            "weights_mode": dl.get("weights_mode"),
            "avg_inference_ms": dl.get("avg_inference_ms"),
        },
    }


@app.post("/api/embedding/export")
async def export_single_video_embedding_post(req: SingleVideoEmbeddingExportRequest):
    """
    POST /api/embedding/export:
    Exports single cattle video 512-D ArcFace/GeM Biometric Embedding vector.
    """
    return await _handle_single_video_embedding_export(
        video_path=req.video_path,
        pool_method=req.pool_method or "median",
        req_opts=req,
    )


@app.get("/api/embedding/export")
async def export_single_video_embedding_get(
    video_path: str = Query(..., description="Path or filename of video"),
    pool_method: Optional[str] = Query("median", description="'median' or 'mean'"),
):
    """
    GET /api/embedding/export:
    Exports single cattle video 512-D ArcFace/GeM Biometric Embedding vector.
    """
    return await _handle_single_video_embedding_export(
        video_path=video_path,
        pool_method=pool_method or "median",
    )


@app.get("/api/video/match-config")
@app.get("/api/video/matcher-config")
async def get_spatial_match_config_endpoint():
    """Return the dedicated spatial matcher calibration state, never legacy calibration."""
    params_path = BASE_DIR / "config" / "matching_params.json"
    threshold_path = BASE_DIR / "config" / "matching_threshold.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    calibrated = False
    threshold = None
    try:
        config = json.loads(threshold_path.read_text(encoding="utf-8"))
        value = float(config["threshold"])
        calibrated = bool(config.get("calibrated")) and int(config.get("genuine_pairs", 0)) >= 30 and int(config.get("impostor_pairs", 0)) >= 30
        if calibrated:
            threshold = value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    return {
        "calibrated": calibrated,
        "status": "CALIBRATED" if calibrated else "UNCALIBRATED",
        "threshold": threshold,
        "dev_provisional_threshold": float(params["dev_provisional_threshold"]),
        "min_calibration_genuine_pairs": int(params["min_calibration_genuine_pairs"]),
        "min_calibration_impostor_pairs": int(params["min_calibration_impostor_pairs"]),
    }

get_matcher_config_endpoint = get_spatial_match_config_endpoint


class VideoMatchRequest(BaseModel):
    video_a_path: Optional[str] = None
    video_b_path: Optional[str] = None
    filter_method: Optional[str] = "frangi"
    threshold_method: Optional[str] = "adaptive"
    adaptive_block_size: Optional[int] = 15
    adaptive_c: Optional[int] = 2
    exclude_nostrils: Optional[bool] = True
    min_spur_length: Optional[int] = 8
    min_component_size: Optional[int] = 30
    frames_a: Optional[List[Any]] = None
    frames_b: Optional[List[Any]] = None
    cache_id_a: Optional[str] = None
    cache_id_b: Optional[str] = None
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6


@app.post("/api/video/match")
async def match_video_endpoint(req: VideoMatchRequest):
    """Match two video identifiers through a bounded cache and the spatial matcher, or match pre-extracted frame sets."""
    if req.frames_a and req.frames_b:
        try:
            fn = partial(match_muzzle_frame_sets, frames_a=req.frames_a, frames_b=req.frames_b, output_dir=OUTPUT_DIR)
            return await asyncio.to_thread(fn)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Frame set matching failed: {exc}")

    if not req.video_a_path or not req.video_b_path:
        raise HTTPException(
            status_code=400,
            detail="Both video_a_path and video_b_path must be provided.",
        )
    va = resolve_video_path(req.video_a_path)
    vb = resolve_video_path(req.video_b_path)
    if not va or not va.exists():
        raise HTTPException(status_code=404, detail=f"Video A not found: {req.video_a_path}")
    if not vb or not vb.exists():
        raise HTTPException(status_code=404, detail=f"Video B not found: {req.video_b_path}")

    options = (
        req.filter_method or "frangi",
        req.threshold_method or "adaptive",
        int(req.adaptive_block_size or 15),
        int(req.adaptive_c or 2),
        bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
        int(req.min_spur_length or 8),
        int(req.min_component_size or 30),
        req.device or "cpu",
        int(req.cpu_cores or 6),
    )
    keys = [
        _spatial_cache_key(path, *options) for path in (va, vb)
    ]
    frame_sets = [_get_extraction_cache(key) for key in keys]

    async def extract_missing(path, key):
        async with _EXTRACTION_LOCK:
            result = await asyncio.to_thread(_extract_video_for_spatial_match, path, req)
            frames = result.get("frames") or []
            _put_extraction_cache(key, frames)
            return frames

    try:
        if frame_sets[0] is None:
            frame_sets[0] = await extract_missing(va, keys[0])
        if frame_sets[1] is None:
            frame_sets[1] = await extract_missing(vb, keys[1])
        if not frame_sets[0] or not frame_sets[1]:
            raise HTTPException(status_code=422, detail="One or both videos produced no usable frames.")

        cores = int(req.cpu_cores or 6)
        result = await asyncio.to_thread(match_spatial_video_sets, frame_sets[0], frame_sets[1], cpu_cores=cores)
        # The matcher has already converted its visualization to base64. Never serialize ndarrays.
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spatial video matching failed: {exc}")


@app.post("/api/video/match-legacy")
async def match_video_legacy_endpoint(req: VideoMatchRequest):
    """Compatibility endpoint for the previous SIFT/minutiae API."""
    loop = asyncio.get_event_loop()
    from functools import partial
    if req.frames_a and req.frames_b:
        try:
            fn = partial(match_muzzle_frame_sets, frames_a=req.frames_a, frames_b=req.frames_b, output_dir=OUTPUT_DIR)
            return await loop.run_in_executor(None, fn)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Frame set matching failed: {exc}")
    if req.video_a_path and req.video_b_path:
        va, vb = resolve_video_path(req.video_a_path), resolve_video_path(req.video_b_path)
        if not va or not va.exists():
            raise HTTPException(status_code=404, detail=f"Video A not found: {req.video_a_path}")
        if not vb or not vb.exists():
            raise HTTPException(status_code=404, detail=f"Video B not found: {req.video_b_path}")
        fn = partial(match_two_videos_end_to_end,
            video_a_path=str(va), video_b_path=str(vb),
            filter_method=req.filter_method or "frangi", threshold_method=req.threshold_method or "adaptive",
            adaptive_block_size=int(req.adaptive_block_size or 15), adaptive_c=int(req.adaptive_c or 2),
            exclude_nostrils=bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
            device=req.device or "cpu", cpu_cores=int(req.cpu_cores or 6), output_dir=OUTPUT_DIR)
        try:
            async with _EXTRACTION_LOCK:
                return await asyncio.to_thread(fn)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Legacy video matching failed: {exc}")
    raise HTTPException(status_code=400, detail="Legacy endpoint requires video paths or pre-extracted frame sets.")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
