"""
Muzzle Detection, Segmentation, and Benchmarking Streaming Routes
=================================================================
SSE streaming endpoints for Mask R-CNN detection, image/video muzzle cropping,
and comparative hardware benchmarks.
"""

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from ..helpers import (
    parse_cpu_cores,
    stream_subprocess_output,
    resolve_video_path,
    OUTPUT_DIR,
    TEST_SAMPLES_DIR,
    PYTHON_EXEC,
    CROPPER_SCRIPT,
    MATCHER_SCRIPT,
    WEIGHTS_PATH,
)

router = APIRouter()


@router.get("/api/stream-video")
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
    if os.path.getsize(video_path) == 0:
        raise HTTPException(status_code=400, detail="Video file is empty (0 bytes). Please select a valid video file.")

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
        env_extra["TF_NUM_INTEROP_THREADS"] = str(min(parsed_cores, 2))
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


@router.get("/api/stream-image")
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
        env_extra["TF_NUM_INTEROP_THREADS"] = str(min(parsed_cores, 2))
        env_extra["MKL_NUM_THREADS"] = str(parsed_cores)
        env_extra["OPENBLAS_NUM_THREADS"] = str(parsed_cores)

    if square:
        cmd.append("--square")
    if fast_mode is True:
        cmd.append("--fast-mode")
    elif fast_mode is False:
        cmd.append("--no-fast-mode")

    return StreamingResponse(stream_subprocess_output(cmd, env_extra=env_extra), media_type="text/event-stream")


@router.get("/api/stream-benchmark")
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


@router.get("/api/stream-compare-videos")
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
    if resolved_a.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video A file is empty (0 bytes): {resolved_a.name}")
    if resolved_b.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video B file is empty (0 bytes): {resolved_b.name}")

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
        env_extra["TF_NUM_INTEROP_THREADS"] = str(min(parsed_cores, 2))
        env_extra["MKL_NUM_THREADS"] = str(parsed_cores)
        env_extra["OPENBLAS_NUM_THREADS"] = str(parsed_cores)

    return StreamingResponse(stream_subprocess_output(cmd, env_extra=env_extra), media_type="text/event-stream")
