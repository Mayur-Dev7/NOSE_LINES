"""
Biometric Matching & Cosine Similarity Routes
=============================================
Endpoints for topological groove graph parameters, ConvNeXt-Tiny & ResNet-50
Deep Learning Biometric Embeddings, SIFT/RANSAC keypoint matching, and secondary
ORB keypoint verification.
"""

import os
import json
import time
import math
import asyncio
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..helpers import (
    BASE_DIR,
    OUTPUT_DIR,
    WEIGHTS_PATH,
    resolve_video_path,
    _EXTRACTION_LOCK,
    _spatial_cache_key,
    _put_extraction_cache,
    _get_extraction_cache,
    _json_safe_frame,
    _sanitize_floats,
    _extract_video_for_spatial_match,
)
from ml.video_groove_extractor import (
    extract_five_frames_groove_features,
    compute_cosine_similarity,
)
from ml.muzzle_matcher import (
    enhance_muzzle_image,
    compare_two_muzzles,
    match_muzzle_frame_sets,
    match_two_videos_end_to_end,
    match_video_sets as match_spatial_video_sets,
)

router = APIRouter()


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


@router.post("/api/video/extract-five-muzzle-parameters")
async def extract_video_five_frames_endpoint(req: ExtractVideoFiveFramesRequest):
    """
    Extracts 5 muzzle images, ten topological parameters, and 512-D Deep Learning Biometric Embeddings.
    The backend caches internal NumPy arrays for spatial matching and strips those arrays from the HTTP response.
    """
    resolved_v = resolve_video_path(req.video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")
    if resolved_v.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video file is empty (0 bytes): {resolved_v.name}")

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
        return _sanitize_floats(safe)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
    dl_match_threshold: Optional[float] = None
    dl_likely_threshold: Optional[float] = None
    graph_match_threshold: Optional[float] = None
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6
    crop_a_path: Optional[str] = None
    crop_b_path: Optional[str] = None
    frames_a: Optional[List[Dict[str, Any]]] = None
    frames_b: Optional[List[Dict[str, Any]]] = None
    orb_threshold: Optional[int] = 35


def execute_secondary_orb_verification(
    crop_a_path: Optional[str] = None,
    crop_b_path: Optional[str] = None,
    frames_a: Optional[List[Dict[str, Any]]] = None,
    frames_b: Optional[List[Dict[str, Any]]] = None,
    threshold: int = 35,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes secondary classical ORB keypoint & Lowe's ratio matching (Mask_RCNN / statusprint.py method).
    Compares the highest quality or canonically aligned muzzle crops between two cattle video sets.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / f"secondary_orb_{int(time.time()*1000)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_a = []
    if crop_a_path and Path(crop_a_path).exists():
        cand_a.append(Path(crop_a_path))
    if frames_a:
        for f in frames_a:
            p_enh = f.get("enhanced_path")
            p_crop = f.get("crop_path")
            if p_enh and Path(p_enh).exists():
                cand_a.append(Path(p_enh))
            elif p_crop and Path(p_crop).exists():
                cand_a.append(Path(p_crop))

    cand_b = []
    if crop_b_path and Path(crop_b_path).exists():
        cand_b.append(Path(crop_b_path))
    if frames_b:
        for f in frames_b:
            p_enh = f.get("enhanced_path")
            p_crop = f.get("crop_path")
            if p_enh and Path(p_enh).exists():
                cand_b.append(Path(p_enh))
            elif p_crop and Path(p_crop).exists():
                cand_b.append(Path(p_crop))

    if not cand_a or not cand_b:
        return {
            "executed": False,
            "error": "No valid crop image files found for Video A or Video B",
            "good_matches": 0,
            "threshold": threshold,
            "is_match": False,
            "verdict": "UNAVAILABLE",
        }

    best_res = None
    best_matches = -1
    best_pair_paths = (None, None)

    for pa in cand_a[:3]:
        img_a = cv2.imread(str(pa))
        if img_a is None:
            continue
        if img_a.shape[:2] == (448, 448):
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
            up_a = img_a if img_a.ndim == 3 else cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
        else:
            gray_a, up_a = enhance_muzzle_image(img_a, method="lanczos4", scale_factor=2, unsharp=True, clahe=True)

        for pb in cand_b[:3]:
            img_b = cv2.imread(str(pb))
            if img_b is None:
                continue
            if img_b.shape[:2] == (448, 448):
                gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
                up_b = img_b if img_b.ndim == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
            else:
                gray_b, up_b = enhance_muzzle_image(img_b, method="lanczos4", scale_factor=2, unsharp=True, clahe=True)

            res = compare_two_muzzles(
                gray1=gray_a,
                gray2=gray_b,
                nfeatures=1000,
                threshold=threshold,
                bgr1=up_a,
                bgr2=up_b,
                save_visual_path=None,
            )
            if res["good_matches"] > best_matches:
                best_matches = res["good_matches"]
                best_res = res
                best_pair_paths = (pa, pb)

    if best_res is None:
        best_res = {
            "good_matches": 0,
            "threshold": threshold,
            "is_match": False,
            "verdict": "DIFFERENT_CATTLE",
            "similarity_percent": 0.0,
            "kp1_count": 0,
            "kp2_count": 0,
            "visual_path": None,
        }

    visual_diagram_path = output_dir / "secondary_orb_match_diagram.jpg"
    if best_pair_paths[0] and best_pair_paths[1]:
        img_a = cv2.imread(str(best_pair_paths[0]))
        img_b = cv2.imread(str(best_pair_paths[1]))
        if img_a is not None and img_b is not None:
            if img_a.shape[:2] == (448, 448):
                gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
                up_a = img_a if img_a.ndim == 3 else cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
            else:
                gray_a, up_a = enhance_muzzle_image(img_a, method="lanczos4", scale_factor=2, unsharp=True, clahe=True)

            if img_b.shape[:2] == (448, 448):
                gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
                up_b = img_b if img_b.ndim == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
            else:
                gray_b, up_b = enhance_muzzle_image(img_b, method="lanczos4", scale_factor=2, unsharp=True, clahe=True)

            best_res = compare_two_muzzles(
                gray1=gray_a,
                gray2=gray_b,
                nfeatures=1000,
                threshold=threshold,
                bgr1=up_a,
                bgr2=up_b,
                save_visual_path=str(visual_diagram_path),
            )

    visual_url = None
    if visual_diagram_path.exists():
        rel = visual_diagram_path.relative_to(OUTPUT_DIR)
        visual_url = f"/output/{rel}"

    return {
        "executed": True,
        "method_name": "ORB Keypoints + Lowe's Ratio Hamming Matcher (statusprint.py / Mask_RCNN Method)",
        "good_matches": int(best_res["good_matches"]),
        "threshold": int(threshold),
        "is_match": bool(best_res["is_match"]),
        "verdict": best_res["verdict"],
        "similarity_percent": float(best_res["similarity_percent"]),
        "kp1_count": int(best_res.get("kp1_count", 0)),
        "kp2_count": int(best_res.get("kp2_count", 0)),
        "visual_path": str(visual_diagram_path) if visual_diagram_path.exists() else None,
        "visual_url": visual_url,
        "weights_used": str(WEIGHTS_PATH),
        "weights_verified": WEIGHTS_PATH.exists(),
        "tested_pair": {
            "crop_a": best_pair_paths[0].name if best_pair_paths[0] else None,
            "crop_b": best_pair_paths[1].name if best_pair_paths[1] else None,
        }
    }


@router.post("/api/video/cosine-similarity")
@router.post("/api/video/dl-cosine-similarity")
async def compute_cosine_similarity_endpoint(req: CosineSimilarityRequest):
    """
    Computes Normalized Cosine Similarity on classical 10-D graph minutiae parameters
    (with Bray-Curtis magnitude penalty calibration) and Deep Learning Biometric
    Embeddings (ConvNeXt-Tiny / ResNet-50 Spatial GeM & ArcFace).
    Supports dual embeddings (mean and median pooling) with full backward compatibility.
    When primary model indicates a match OR is inconclusive, executes secondary confirmation
    via Mask_RCNN/statusprint.py ORB keypoint Lowe's ratio matcher.
    """
    t0 = time.perf_counter()
    try:
        kwargs = {}
        if req.dl_match_threshold is not None:
            kwargs["dl_match_threshold"] = float(req.dl_match_threshold)
        if req.dl_likely_threshold is not None:
            kwargs["dl_likely_threshold"] = float(req.dl_likely_threshold)
        if req.graph_match_threshold is not None:
            kwargs["graph_match_threshold"] = float(req.graph_match_threshold)

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
            **kwargs,
        )

        # Secondary verification logic from /home/adis_mayur/F_CODES/Mask_RCNN
        dl_sim = float(res.get("dl_similarity_pct") or res.get("median_pooling_similarity_pct") or res.get("cosine_similarity", 0.0))
        dl_verdict = str(res.get("dl_verdict") or res.get("verdict", ""))
        match_thresh = float(req.dl_match_threshold or 70.0)
        likely_thresh = float(req.dl_likely_threshold or 58.0)
        is_dl_match = (dl_sim >= match_thresh) and ("DISTINCT" not in dl_verdict.upper())

        is_not_confident = (
            "INCONCLUSIVE" in dl_verdict.upper()
            or "LIKELY" in dl_verdict.upper()
            or "PARTIAL" in dl_verdict.upper()
            or (likely_thresh <= dl_sim < match_thresh)
            or (45.0 <= dl_sim < match_thresh)
        )

        should_run_secondary = is_dl_match or is_not_confident
        has_crops = bool(req.crop_a_path or req.crop_b_path or req.frames_a or req.frames_b)

        secondary_info = None
        if should_run_secondary and has_crops:
            trigger_reason = (
                "CONFIRMED_MATCH_VERIFICATION" if is_dl_match
                else "INCONCLUSIVE_ARBITRATION"
            )
            orb_out = execute_secondary_orb_verification(
                crop_a_path=req.crop_a_path,
                crop_b_path=req.crop_b_path,
                frames_a=req.frames_a,
                frames_b=req.frames_b,
                threshold=req.orb_threshold or 35,
            )
            orb_out["trigger_reason"] = trigger_reason

            orb_match = orb_out.get("is_match", False)
            if is_dl_match and orb_match:
                consensus_verdict = "CONFIRMED MATCH: Verified by both Deep ArcFace Embedding & ORB Keypoints"
                consensus_badge = "emerald"
                consensus_code = "FULL_MATCH_AGREEMENT"
            elif is_dl_match and not orb_match:
                consensus_verdict = "HIGH EMBEDDING SIMILARITY (ORB Keypoints < 35: Viewpoint Shift or Surface Deformation)"
                consensus_badge = "teal"
                consensus_code = "DEEP_MATCH_ORB_PENDING"
            elif is_not_confident and orb_match:
                consensus_verdict = "ARBITRATED MATCH: Primary Embedding Borderline, Confirmed Same Cattle by ORB Keypoints"
                consensus_badge = "emerald"
                consensus_code = "ORB_ARBITRATED_MATCH"
            elif is_not_confident and not orb_match:
                consensus_verdict = "DISTINCT ANIMAL: Primary Embedding Borderline, Confirmed Different by ORB Keypoints"
                consensus_badge = "rose"
                consensus_code = "ORB_ARBITRATED_DISTINCT"
            else:
                consensus_verdict = "DISTINCT: Both Methods Confirm Different Animals"
                consensus_badge = "rose"
                consensus_code = "FULL_DISTINCT"

            secondary_info = {
                "triggered": True,
                "trigger_reason": trigger_reason,
                "consensus_verdict": consensus_verdict,
                "consensus_badge": consensus_badge,
                "consensus_code": consensus_code,
                "primary_model": {
                    "name": "ConvNeXt-Tiny + SpatialPyramidGeM (ArcFace 512-D)",
                    "similarity_pct": dl_sim,
                    "verdict": dl_verdict,
                    "is_match": is_dl_match,
                },
                "secondary_model": {
                    "name": "Mask R-CNN + Lanczos-4 Sinc + CLAHE + ORB Lowe's Ratio (τ = 35)",
                    "good_matches": orb_out["good_matches"],
                    "threshold": orb_out["threshold"],
                    "verdict": orb_out["verdict"],
                    "is_match": orb_out["is_match"],
                    "similarity_percent": orb_out["similarity_percent"],
                    "visual_url": orb_out["visual_url"],
                    "visual_path": orb_out["visual_path"],
                    "tested_pair": orb_out.get("tested_pair", {}),
                },
                "raw_orb_output": orb_out,
            }
        else:
            secondary_info = {
                "triggered": False,
                "trigger_reason": "NOT_TRIGGERED_PRIMARY_MODEL_CONFIDENT_DISTINCT" if not should_run_secondary else "NO_CROPS_PROVIDED",
            }

        res["secondary_verification"] = secondary_info

        t_sim_ms = (time.perf_counter() - t0) * 1000
        if "step_timings" not in res or not isinstance(res["step_timings"], dict):
            res["step_timings"] = {}
        res["step_timings"]["step_7_consensus_ms"] = max(0.5, round(t_sim_ms, 1))
        res["step_timings"]["t_consensus_ms"] = res.get("t_consensus_ms", max(0.5, round(t_sim_ms, 1)))
        res["step_timings"]["total_ms"] = max(0.5, round(t_sim_ms, 1))
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cosine similarity calculation failed: {exc}")


@router.get("/api/video/match-config")
@router.get("/api/video/matcher-config")
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


@router.post("/api/video/match")
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
    if va.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video A is empty (0 bytes): {va.name}")
    if vb.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Video B is empty (0 bytes): {vb.name}")

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


@router.post("/api/video/match-legacy")
async def match_video_legacy_endpoint(req: VideoMatchRequest):
    """Compatibility endpoint for the previous SIFT/minutiae API."""
    loop = asyncio.get_event_loop()
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
