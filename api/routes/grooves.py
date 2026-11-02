"""
Muzzle Groove & Skeleton Extraction Routes
==========================================
Interactive tuning and video batch processing endpoints for cattle muzzle dark groove
extraction, Frangi/Blackhat filtering, Otsu/Adaptive binarization, and skeletonization.
"""

import time
import asyncio
from functools import partial
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..helpers import resolve_image_path, resolve_video_path, OUTPUT_DIR
from ml.groove_extractor import extract_muzzle_grooves_pipeline

try:
    from tools.process_video_grooves import process_video_grooves
except ImportError:
    try:
        from process_video_grooves import process_video_grooves
    except ImportError:
        from ml.process_video_grooves import process_video_grooves

router = APIRouter()


class GrooveTuneRequest(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    method: str = "blackhat"
    scales: List[int] = Field(default_factory=lambda: [9, 15, 21])
    threshold_method: str = "otsu"
    min_component_size: int = 15
    skeleton_enabled: bool = True


@router.post("/api/grooves/tune")
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


@router.get("/api/grooves/sample-result")
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


@router.post("/api/grooves/process-video")
async def process_video_grooves_endpoint(req: ProcessVideoGroovesRequest):
    """Processes a cattle video end-to-end to extract muzzle grooves and skeletons."""
    resolved_v = resolve_video_path(req.video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")
    if resolved_v.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video file is empty (0 bytes): {resolved_v.name}")

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
