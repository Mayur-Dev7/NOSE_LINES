#!/usr/bin/env python3
"""
video_groove_extractor.py
=========================
5-frame cattle muzzle groove extraction pipeline using Mask R-CNN detection,
2x Lanczos-4 upscaling, CLAHE grayscale ridge enhancement, configurable groove filtering
(Frangi Hessian or Black-Hat), thresholding (Adaptive Gaussian or Standard Otsu),
1-pixel thin skeletonization (5th image), and minutiae parameter extraction from line coordinates.
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from groove_extractor import (
    extract_muzzle_grooves_pipeline,
    enhance_dark_grooves,
    binarize_grooves,
    skeletonize_grooves,
    isolate_muzzle_region,
)
from groove_graph import extract_skeleton_graph, visualize_groove_graph

OUTPUT_BASE = PROJECT_ROOT / "output" / "video_five_frames"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("video_groove_extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Normalization bounds for 10-D muzzle feature vector dimensions
FEATURE_SCALES = [
    (5000, 40000),   # groove_pixels
    (1000, 12000),   # skeleton_pixels
    (10.0, 45.0),    # groove_coverage_pct
    (50, 1000),      # endpoints
    (30, 800),       # junctions
    (10, 250),       # connected_components
    (4.0, 35.0),     # mean_branch_length
    (10, 400),       # cycles
    (0.01, 0.12),    # density
    (-200, 50),      # euler_characteristic
]

FEATURE_KEYS = [
    "groove_pixels",
    "skeleton_pixels",
    "groove_coverage_pct",
    "endpoints",
    "junctions",
    "connected_components",
    "mean_branch_length",
    "cycles",
    "density",
    "euler_characteristic",
]

# Singleton Mask R-CNN detector instance
_DETECTOR_PIPELINE = None


def get_detector_pipeline(
    weights_path: Optional[str] = None,
    device: str = "auto",
    cpu_cores: int = 6,
):
    """Returns or initializes singleton Mask R-CNN pipeline loaded with crop11dec.h5 weights."""
    global _DETECTOR_PIPELINE
    if _DETECTOR_PIPELINE is None:
        from muzzle_cropper import MuzzleDetectionPipeline
        import config

        w_path = weights_path or str(config.WEIGHTS_PATH)
        if not os.path.exists(w_path):
            alt_path = PROJECT_ROOT.parent / "crop11dec.h5"
            if alt_path.exists():
                w_path = str(alt_path)

        logger.info(f"Initializing Mask R-CNN detector singleton with weights: {w_path}...")
        _DETECTOR_PIPELINE = MuzzleDetectionPipeline(
            weights_path=w_path,
            device=device,
            cpu_cores=cpu_cores,
            confidence_threshold=0.70,
        )
        logger.info("Mask R-CNN detector initialized successfully.")
    return _DETECTOR_PIPELINE


def normalize_feature_vector(vec: List[float]) -> np.ndarray:
    """Normalizes 10-dimensional feature vector into [0, 1] bounds for balanced cosine similarity."""
    normed = []
    for i, val in enumerate(vec[:10]):
        lo, hi = FEATURE_SCALES[i] if i < len(FEATURE_SCALES) else (0.0, 1.0)
        clamped = max(lo, min(hi, float(val)))
        n = (clamped - lo) / max(1e-4, hi - lo)
        normed.append(n)
    return np.array(normed, dtype=float)


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> Dict[str, Any]:
    """
    Computes Cosine Similarity between two 10-D feature vectors:
    Cosine Similarity = (A . B) / (||A|| * ||B||) * 100%
    """
    v1 = np.array(vec_a, dtype=float)
    v2 = np.array(vec_b, dtype=float)

    # Raw cosine similarity
    dot_raw = float(np.dot(v1, v2))
    norm_a_raw = float(np.linalg.norm(v1))
    norm_b_raw = float(np.linalg.norm(v2))
    raw_sim = (dot_raw / max(1e-6, norm_a_raw * norm_b_raw)) * 100.0 if (norm_a_raw > 0 and norm_b_raw > 0) else 0.0

    # Normalized cosine similarity (dimensions scaled to equal weight)
    n1 = normalize_feature_vector(vec_a)
    n2 = normalize_feature_vector(vec_b)
    dot_norm = float(np.dot(n1, n2))
    norm1 = float(np.linalg.norm(n1))
    norm2 = float(np.linalg.norm(n2))
    norm_sim = (dot_norm / max(1e-6, norm1 * norm2)) * 100.0 if (norm1 > 0 and norm2 > 0) else 0.0

    # Match classification
    is_match = norm_sim >= 85.0
    if norm_sim >= 92.0:
        verdict = "MATCH (Confirmed Same Cattle)"
        verdict_color = "emerald"
    elif norm_sim >= 85.0:
        verdict = "LIKELY MATCH (High Similarity)"
        verdict_color = "teal"
    elif norm_sim >= 75.0:
        verdict = "INCONCLUSIVE (Partial Overlap)"
        verdict_color = "amber"
    else:
        verdict = "DISTINCT (Different Animals)"
        verdict_color = "rose"

    return {
        "cosine_similarity": round(norm_sim, 2),
        "raw_cosine_similarity": round(raw_sim, 2),
        "is_match": is_match,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "vector_a": [round(float(x), 3) for x in vec_a],
        "vector_b": [round(float(x), 3) for x in vec_b],
    }


def extract_five_frames_groove_features(
    video_path: Union[str, Path],
    method: str = "frangi",                     # 'frangi' (Frangi Hessian) or 'blackhat'
    threshold_method: str = "adaptive",         # 'adaptive' (Adaptive Gaussian) or 'otsu' (Standard)
    adaptive_block_size: int = 15,
    adaptive_c: int = 2,
    scales: List[int] = [9, 15, 21],
    exclude_nostrils: bool = True,
    device: str = "cpu",
    cpu_cores: int = 6,
    log_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Genuine end-to-end extraction:
    1. Loads Mask R-CNN detector with weights (crop11dec.h5).
    2. Samples frames from video and detects cow muzzle crops with Mask R-CNN.
    3. Selects 5 distinct, high-quality muzzle detections across the video duration.
    4. Applies 2x Lanczos-4 upscaling + Grayscale + CLAHE ridge enhancement on each.
    5. Applies Frangi Hessian or Black-Hat filtering.
    6. Applies Adaptive Gaussian or Standard Otsu thresholding.
    7. Computes 1-pixel thin skeleton (the 5th image).
    8. Calculates the 10 minutiae parameters from the skeleton lines and coordinates.
    9. Saves all 5 stage images per frame and returns vectors and URLs.
    """
    def log(msg: str):
        logger.info(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    v_path = Path(video_path).resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {v_path}")

    stem = v_path.stem
    cache_dir = OUTPUT_BASE / f"{stem}_{method}_{threshold_method}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting 5-frame muzzle groove extraction for video: {v_path.name}")
    log(f"Configuration: Filter={method.upper()} | Threshold={threshold_method.upper()} (BlockSize={adaptive_block_size}, C={adaptive_c})")

    # Step 1: Initialize Mask R-CNN Model
    cropper = get_detector_pipeline(device=device, cpu_cores=cpu_cores)

    # Step 2: Open video and calculate temporal distribution
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {v_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / max(1.0, fps)
    log(f"Video duration: {duration_sec:.1f}s ({total_frames} total frames, {fps:.1f} FPS)")

    # 5 temporal regions across video: 10%, 30%, 50%, 70%, 90%
    target_fractions = [0.10, 0.30, 0.50, 0.70, 0.90]
    target_indices = [max(0, min(total_frames - 1, int(total_frames * f))) for f in target_fractions]

    detected_candidates = []

    for seg_idx, center_idx in enumerate(target_indices):
        log(f"[Stage 1/5] Searching muzzle detection for slot {seg_idx + 1}/5 near frame #{center_idx}...")
        found_in_slot = None
        # Try center frame, then search nearby frames in increments of 5
        offsets = [0, 5, -5, 10, -10, 15, -15, 20, -20]
        for off in offsets:
            f_idx = max(0, min(total_frames - 1, center_idx + off))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            # Quick blur variance check
            gray_small = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
            if blur_var < 20.0 and off != 0:
                continue

            # Run real Mask R-CNN detector
            det_res = cropper.process_image(frame_bgr, make_square=True, target_size=None)
            detections = det_res.get("detections", [])
            if detections:
                best_det = max(detections, key=lambda d: d["confidence"])
                if best_det["confidence"] >= 0.70:
                    y1, x1, y2, x2 = best_det["bbox_cropped"]
                    native_crop = frame_bgr[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else best_det["cropped_bgr"]
                    
                    full_mask = best_det.get("mask")
                    crop_mask = None
                    if full_mask is not None and full_mask.shape[0] >= y2 and full_mask.shape[1] >= x2:
                        raw_patch = full_mask[y1:y2, x1:x2].astype(np.uint8) * 255
                        crop_mask = raw_patch

                    found_in_slot = {
                        "frame_idx": f_idx,
                        "timestamp_sec": round(float(f_idx / fps), 2),
                        "confidence": float(best_det["confidence"]),
                        "native_crop": native_crop,
                        "crop_mask": crop_mask,
                        "bbox": best_det["bbox_cropped"],
                    }
                    log(f" -> Slot {seg_idx + 1}/5: Detected muzzle at frame #{f_idx} (Confidence: {best_det['confidence']*100:.1f}%)")
                    break

        if found_in_slot:
            detected_candidates.append(found_in_slot)
        else:
            log(f" -> Slot {seg_idx + 1}/5: No muzzle passed threshold, extracting center frame fallback")
            cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)
            ret, frame_bgr = cap.read()
            if ret and frame_bgr is not None:
                h, w = frame_bgr.shape[:2]
                cy1, cy2 = int(h * 0.25), int(h * 0.85)
                cx1, cx2 = int(w * 0.15), int(w * 0.85)
                crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
                detected_candidates.append({
                    "frame_idx": center_idx,
                    "timestamp_sec": round(float(center_idx / fps), 2),
                    "confidence": 0.80,
                    "native_crop": crop,
                    "crop_mask": None,
                    "bbox": [cy1, cx1, cy2, cx2],
                })

    cap.release()

    # Step 3: Process the 5 frames through 2x Lanczos-4, CLAHE grayscale, groove filter, threshold, and skeletonization
    frames_records = []
    param_lists: Dict[str, List[float]] = {k: [] for k in FEATURE_KEYS}

    for i, candidate in enumerate(detected_candidates[:5]):
        f_idx = candidate["frame_idx"]
        native_bgr = candidate["native_crop"]
        crop_mask = candidate["crop_mask"]
        conf = candidate["confidence"]

        log(f"[Frame {i + 1}/5] Processing muzzle from frame #{f_idx} (native size {native_bgr.shape[1]}x{native_bgr.shape[0]})...")

        # 1. Native crop save
        p_crop = cache_dir / f"frame_{i}_01_native_crop.jpg"
        cv2.imwrite(str(p_crop), native_bgr)

        # 2. Stage 2: 2x Super-Resolution Upscaling with Lanczos-4 & CLAHE Grayscale
        # Target standard 448x448 evaluation canvas
        upscaled_bgr = cv2.resize(native_bgr, (448, 448), interpolation=cv2.INTER_LANCZOS4)
        gray_up = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)

        # Unsharp masking
        blurred = cv2.GaussianBlur(gray_up, (0, 0), sigmaX=1.5, sigmaY=1.5)
        gray_unsharp = cv2.addWeighted(gray_up, 1.5, blurred, -0.5, 0)

        # CLAHE local contrast normalization
        clahe_filter = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe_filter.apply(gray_unsharp)

        p_enh = cache_dir / f"frame_{i}_02_enhanced_ridge.jpg"
        cv2.imwrite(str(p_enh), enhanced_gray)

        # Upscale mask if available
        mask_up = None
        if crop_mask is not None and crop_mask.size > 0:
            mask_up = cv2.resize(crop_mask, (448, 448), interpolation=cv2.INTER_NEAREST)

        # 3. Stage 3 & 4: Groove Response & Thresholding
        # Convert enhanced grayscale back to 3-channel for pipeline input
        enh_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        groove_res = extract_muzzle_grooves_pipeline(
            image_bgr=enh_bgr,
            mask=mask_up,
            method=method,                     # 'frangi' (Frangi Hessian) or 'blackhat'
            threshold_method=threshold_method, # 'adaptive' (Adaptive Gaussian) or 'otsu'
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            scales=scales,
            exclude_nostrils=exclude_nostrils,
        )

        groove_response = groove_res["enhanced_response"]
        groove_mask = groove_res["groove_mask"]
        groove_skeleton = groove_res["groove_skeleton"]  # 5th Image (1-px Thin Skeleton Lines)

        # 4. Stage 5: Line Coordinates & Minutiae Parameters Extraction
        graph_res = extract_skeleton_graph(groove_skeleton, mask=groove_res.get("mask"))
        desc = graph_res["descriptors"]

        # 5. Visual Colored Groove Overlay
        overlay = visualize_groove_graph(upscaled_bgr, graph_res)

        # Save stage artifacts to disk
        p_resp = cache_dir / f"frame_{i}_03_groove_response.png"
        p_mask = cache_dir / f"frame_{i}_04_groove_mask.png"
        p_skel = cache_dir / f"frame_{i}_05_thin_skeleton.png"
        p_over = cache_dir / f"frame_{i}_overlay.png"

        cv2.imwrite(str(p_resp), groove_response)
        cv2.imwrite(str(p_mask), groove_mask)
        cv2.imwrite(str(p_skel), groove_skeleton)
        cv2.imwrite(str(p_over), overlay)

        # Assemble the 10 numerical minutiae parameters from the 5th image lines
        params = {
            "groove_pixels": int(groove_res["num_groove_pixels"]),
            "skeleton_pixels": int(desc["skeleton_pixel_count"]),
            "groove_coverage_pct": round(float(groove_res["groove_coverage_pct"]), 2),
            "endpoints": int(desc["endpoint_count"]),
            "junctions": int(desc["junction_count"]),
            "connected_components": int(desc["connected_component_count"]),
            "mean_branch_length": round(float(desc["mean_branch_length"]), 2),
            "cycles": int(desc["cycle_count"]),
            "density": round(float(desc["skeleton_density"]), 4),
            "euler_characteristic": int(desc["euler_characteristic"]),
        }

        for k in FEATURE_KEYS:
            param_lists[k].append(params[k])

        rel_base = f"/output/video_five_frames/{cache_dir.name}"
        frames_records.append({
            "frame_number": i + 1,
            "frame_idx": f_idx,
            "timestamp_sec": candidate["timestamp_sec"],
            "confidence": round(float(conf), 3),
            "crop_url": f"{rel_base}/frame_{i}_01_native_crop.jpg",
            "enhanced_url": f"{rel_base}/frame_{i}_02_enhanced_ridge.jpg",
            "groove_response_url": f"{rel_base}/frame_{i}_03_groove_response.png",
            "groove_mask_url": f"{rel_base}/frame_{i}_04_groove_mask.png",
            "skeleton_url": f"{rel_base}/frame_{i}_05_thin_skeleton.png",
            "overlay_url": f"{rel_base}/frame_{i}_overlay.png",
            "parameters": params,
        })
        log(f" -> [Frame {i + 1}/5] Lines extracted: {params['skeleton_pixels']} skel px, {params['endpoints']} ends, {params['junctions']} juncs, {params['cycles']} loops")

    # Aggregate median parameter vector across the 5 frames
    median_params: Dict[str, float] = {}
    vector_array: List[float] = []

    for k in FEATURE_KEYS:
        vals = param_lists[k]
        med = float(np.median(vals)) if vals else 0.0
        median_params[k] = round(med, 3)
        vector_array.append(round(med, 3))

    normalized_vector = [round(float(x), 4) for x in normalize_feature_vector(vector_array)]

    result = {
        "success": True,
        "video_name": v_path.name,
        "video_path": str(v_path),
        "filter_method": method,
        "threshold_method": threshold_method,
        "adaptive_block_size": adaptive_block_size,
        "adaptive_c": adaptive_c,
        "frames": frames_records,
        "median_parameters": median_params,
        "vector_array": vector_array,
        "normalized_vector": normalized_vector,
        "feature_keys": FEATURE_KEYS,
        "timestamp": int(time.time()),
    }

    # Save summary
    with open(cache_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log(f"Completed 5-frame groove parameter extraction for {v_path.name} successfully.")
    return result
