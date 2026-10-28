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
from groove_graph import extract_skeleton_graph, visualize_groove_graph, prune_skeleton

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


def align_muzzle_canonical(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (448, 448),
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Canonically aligns a cropped cattle muzzle image and optional mask to an upright orientation.

    Orientation detection strategy:
      1. Primary: Detect two nostril landmarks (dark regions in the upper 40% of the crop).
         If two nostril centroids are identified (one left, one right), compute the angle
         between them and the horizontal, and rotate so nostrils are level.
      2. Fallback: Compute the mask's principal axis using image central moments
         (angle = 0.5 * atan2(2 * mu11, mu20 - mu02)) to orient symmetrically along the upright axis.
      3. Fallback 2: If neither works, angle is 0.0.

    Transforms:
      - Rotates via cv2.warpAffine with BORDER_REFLECT for image and BORDER_CONSTANT for mask.
      - Resizes to target_size (default 448x448) using cv2.INTER_LANCZOS4 for image and cv2.INTER_NEAREST for mask.

    Returns:
        (aligned_bgr, aligned_mask, angle_deg)
    """
    if image_bgr is None or image_bgr.size == 0:
        empty_bgr = np.zeros((*target_size, 3), dtype=np.uint8)
        empty_mask = np.zeros(target_size, dtype=np.uint8) if mask is not None else None
        return empty_bgr, empty_mask, 0.0

    h, w = image_bgr.shape[:2]
    angle_deg = 0.0
    detected_nostrils = False

    # 1. Primary: Nostril Landmark Detection in upper 40%
    upper_h = max(1, int(h * 0.40))
    upper_roi = image_bgr[:upper_h, :]
    upper_mask = mask[:upper_h, :] if (mask is not None and mask.shape[0] >= upper_h and mask.shape[1] >= w) else None

    gray = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if upper_mask is not None and np.count_nonzero(upper_mask) > 0:
        blurred = np.where(upper_mask > 0, blurred, 255)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(upper_h * w)
    min_area = max(15.0, roi_area * 0.003)
    max_area = roi_area * 0.35

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                cands.append({"area": area, "cx": cx, "cy": cy})

    left_cands = [c for c in cands if c["cx"] < w * 0.55]
    right_cands = [c for c in cands if c["cx"] > w * 0.45]

    if left_cands and right_cands:
        best_left = max(left_cands, key=lambda c: c["area"])
        best_right = max(right_cands, key=lambda c: c["area"])
        dx = best_right["cx"] - best_left["cx"]
        dy = best_right["cy"] - best_left["cy"]
        if dx >= w * 0.15 and abs(dy) < upper_h * 0.6:
            raw_angle = float(np.degrees(np.arctan2(dy, dx)))
            if -45.0 <= raw_angle <= 45.0:
                angle_deg = raw_angle
                detected_nostrils = True

    # 2. Fallback: Principal Axis of Mask using central moments
    if not detected_nostrils:
        if mask is not None and np.count_nonzero(mask) > 50:
            M = cv2.moments(mask.astype(np.uint8))
            mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
            if abs(mu20 - mu02) > 1e-4 or abs(mu11) > 1e-4:
                raw_deg = float(np.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02)))
                # Symmetrically orient along nearest vertical / upright axis in [-45, 45]
                angle_deg = float(((raw_deg + 45.0) % 90.0) - 45.0)

    # 3. Rotate to upright orientation and resize to target_size
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    aligned_bgr = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )

    aligned_mask = None
    if mask is not None:
        aligned_mask = cv2.warpAffine(
            mask,
            rot_mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if (aligned_bgr.shape[1], aligned_bgr.shape[0]) != target_size:
        aligned_bgr = cv2.resize(aligned_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    if aligned_mask is not None and (aligned_mask.shape[1], aligned_mask.shape[0]) != target_size:
        aligned_mask = cv2.resize(aligned_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return aligned_bgr, aligned_mask, round(float(angle_deg), 2)





def canonical_align_muzzle(crop_bgr, mask, target_size=448):
    """Align nostrils using local-background contrast; fall back to the mask principal axis.

    Returns (aligned_bgr, aligned_mask, meta), preserving the legacy align_muzzle_canonical API.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        empty = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        return empty, (np.zeros((target_size, target_size), dtype=np.uint8) if mask is not None else None), {
            "alignment_method": "empty_fallback", "rotation_deg": 0.0
        }
    image = crop_bgr
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    valid = np.full((h, w), 255, dtype=np.uint8)
    if mask is not None and np.asarray(mask).size:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        valid = np.where(m > 0, 255, 0).astype(np.uint8)

    # Difference from a broad local background; this avoids a global percentile
    # and lets small (3-5% of mask) dark nostril cavities remain candidates.
    smooth_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, min(h, w) / 18.0))
    dark_response = cv2.subtract(smooth_bg, gray)
    dark_response[valid == 0] = 0
    roi_h = max(1, int(h * 0.65))
    local = dark_response[:roi_h]
    local_mask = valid[:roi_h]
    nonzero = local[(local_mask > 0) & (local > 0)]
    angle = 0.0
    method = "mask_principal_axis"
    points = []
    if nonzero.size:
        # Robust local-response threshold, independent of the global image histogram.
        threshold = max(8.0, float(np.mean(nonzero) + 0.55 * np.std(nonzero)))
        binary = np.where((local >= threshold) & (local_mask > 0), 255, 0).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask_area = max(1, int(np.count_nonzero(local_mask)))
        candidates = []
        for lab in range(1, count):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            # A nostril can be very small relative to the muzzle; reject specks only.
            if area < max(4, int(mask_area * 0.001)) or area > mask_area * 0.20:
                continue
            cx, cy = centroids[lab]
            if w * 0.12 <= cx <= w * 0.88:
                candidates.append((area, float(cx), float(cy)))
        pairs = []
        for i, left in enumerate(candidates):
            for right in candidates[i + 1:]:
                a, b = sorted((left, right), key=lambda item: item[1])
                dx, dy = b[1] - a[1], b[2] - a[2]
                candidate_angle = math.degrees(math.atan2(dy, dx))
                if dx >= w * 0.16 and abs(candidate_angle) <= 40.0:
                    pairs.append((a[0] + b[0], candidate_angle, a, b))
        if pairs:
            _, raw_angle, left, right = max(pairs, key=lambda pair: pair[0])
            angle = float(raw_angle)
            method = "local_nostril_contrast"
            points = [[left[1], left[2]], [right[1], right[2]]]

    if method != "local_nostril_contrast" and np.count_nonzero(valid) > 50:
        moments = cv2.moments(valid)
        if abs(moments["mu20"] - moments["mu02"]) > 1e-4 or abs(moments["mu11"]) > 1e-4:
            raw = math.degrees(0.5 * math.atan2(2.0 * moments["mu11"], moments["mu20"] - moments["mu02"]))
            angle = float(((raw + 45.0) % 90.0) - 45.0)

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    aligned = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    aligned_mask = cv2.warpAffine(valid, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) if mask is not None else None
    aligned = cv2.resize(aligned, (int(target_size), int(target_size)), interpolation=cv2.INTER_LANCZOS4)
    if aligned_mask is not None:
        aligned_mask = cv2.resize(aligned_mask, (int(target_size), int(target_size)), interpolation=cv2.INTER_NEAREST)
    return aligned, aligned_mask, {
        "alignment_method": method,
        "rotation_deg": round(float(angle), 2),
        "nostril_points": points,
    }

# Optional frame-selection options accepted by the API wrapper; defaults preserve previous callers.
def extract_five_muzzle_stages(
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
    min_spur_length: int = 8,
    min_component_size: int = 30,
) -> Dict[str, Any]:
    """
    Genuine end-to-end extraction with quality gating and canonical alignment:
    1. Loads singleton Mask R-CNN detector with weights (crop11dec.h5).
    2. Samples ~30 candidate frames evenly distributed across video duration.
    3. Evaluates motion blur (Laplacian variance) and detects muzzle with Mask R-CNN.
    4. Applies confidence, Laplacian sharpness, border-margin and mask-area gates from matching_params.json.
    5. Scores valid candidates and selects top_k_frames temporally spread-out frames.
    6. Canonically aligns each crop (nostril leveling / moment symmetry) to 448x448.
    7. Applies CLAHE ridge enhancement on aligned crop.
    8. Applies Frangi Hessian or Black-Hat filtering + Adaptive/Otsu thresholding.
    9. Generates 1-pixel thin skeleton and prunes spurious branches / noise components.
    10. Calculates the 10 minutiae parameters and minutiae point list (x, y, type, orientation).
    11. Saves all 5 stage images per frame and returns vectors and URLs.
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

    matcher_params_path = PROJECT_ROOT / "config" / "matching_params.json"
    try:
        with open(matcher_params_path, "r", encoding="utf-8") as handle:
            matcher_params = json.load(handle)
    except (OSError, json.JSONDecodeError):
        matcher_params = {
            "num_candidate_frames": 30, "top_k_frames": 5,
            "min_detection_confidence": 0.90, "min_mask_area_ratio": 0.05,
            "border_margin_px": 6, "min_laplacian_var": 35.0,
            "canonical_size": 448, "min_spur_length": min_spur_length,
            "min_component_size": min_component_size,
        }

    num_candidate_frames = max(1, int(matcher_params.get("num_candidate_frames", 30)))
    top_k_frames = max(1, int(matcher_params.get("top_k_frames", 5)))
    min_detection_confidence = float(matcher_params.get("min_detection_confidence", 0.90))
    min_mask_area_ratio = float(matcher_params.get("min_mask_area_ratio", 0.05))
    border_margin_px = max(0, int(matcher_params.get("border_margin_px", 6)))
    min_laplacian_var = float(matcher_params.get("min_laplacian_var", 35.0))
    mask_erosion_px = max(0, int(matcher_params.get("mask_erosion_px", 10)))
    canonical_size = max(32, int(matcher_params.get("canonical_size", 448)))

    stem = v_path.stem
    cache_dir = OUTPUT_BASE / f"{stem}_{method}_{threshold_method}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting 5-frame muzzle groove extraction for video: {v_path.name}")
    log(f"Configuration: Filter={method.upper()} | Threshold={threshold_method.upper()} (BlockSize={adaptive_block_size}, C={adaptive_c})")

    # Step 1: Initialize Mask R-CNN Model
    cropper = get_detector_pipeline(device=device, cpu_cores=cpu_cores)

    # Step 2: Open video and calculate temporal candidate frames
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {v_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / max(1.0, fps)
    log(f"Video duration: {duration_sec:.1f}s ({total_frames} total frames, {fps:.1f} FPS)")

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video contains 0 frames: {v_path}")

    # Sample ~30 candidate frames evenly distributed across video duration
    num_samples = min(num_candidate_frames, total_frames)
    candidate_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
    candidate_indices = sorted(list(dict.fromkeys(candidate_indices)))
    log(f"Sampling {len(candidate_indices)} candidate frames across video duration for quality gating...")

    all_candidates: List[Dict[str, Any]] = []

    for f_idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        orig_h, orig_w = frame_bgr.shape[:2]

        # Motion blur check (Laplacian variance)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Run Mask R-CNN detector
        det_res = cropper.process_image(frame_bgr, make_square=True, target_size=None)
        detections = det_res.get("detections", [])

        if not detections:
            cy1, cy2 = int(orig_h * 0.25), int(orig_h * 0.85)
            cx1, cx2 = int(orig_w * 0.15), int(orig_w * 0.85)
            crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
            score = min(100.0, lap_var / 5.0)
            all_candidates.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(float(f_idx / fps), 2),
                "confidence": 0.0,
                "laplacian_var": lap_var,
                "score": score,
                "passed_gating": False,
                "rejection_reasons": ["no_detection"],
                "native_crop": crop,
                "crop_mask": None,
                "bbox": [cy1, cx1, cy2, cx2],
            })
            continue

        best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
        conf = float(best_det.get("confidence", 0.0))

        # Measure sharpness inside the eroded muzzle mask to reduce background influence.
        full_mask_for_quality = best_det.get("mask")
        if full_mask_for_quality is not None and np.asarray(full_mask_for_quality).size:
            quality_mask = np.asarray(full_mask_for_quality)
            if quality_mask.ndim == 3:
                quality_mask = cv2.cvtColor(quality_mask, cv2.COLOR_BGR2GRAY)
            if quality_mask.shape[:2] != gray.shape[:2]:
                quality_mask = cv2.resize(quality_mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            quality_mask = np.where(quality_mask > 0, 255, 0).astype(np.uint8)
            if mask_erosion_px > 0:
                erosion_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * mask_erosion_px + 1, 2 * mask_erosion_px + 1)
                )
                eroded_quality_mask = cv2.erode(quality_mask, erosion_kernel)
            else:
                eroded_quality_mask = quality_mask
            if np.count_nonzero(eroded_quality_mask) >= 64:
                laplacian_map = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_values = laplacian_map[eroded_quality_mask > 0]
                if laplacian_values.size:
                    lap_var = float(np.var(laplacian_values))

        # Quality gating checks:
        # 1. Low confidence (< 0.70)
        # 2. Motion blur (< 25.0)
        # 3. Partial muzzle (bbox margin < 5px if large, or mask area / bbox area < 0.25)
        passed_gating = True
        reasons = []

        if conf < min_detection_confidence:
            passed_gating = False
            reasons.append(f"low_conf={conf:.2f}")

        if lap_var < min_laplacian_var:
            passed_gating = False
            reasons.append(f"blurred={lap_var:.1f}")

        bbox = best_det.get("bbox_original") or best_det.get("bbox_cropped")
        if bbox is not None and len(bbox) == 4:
            y1, x1, y2, x2 = bbox
            if orig_h >= 100 and orig_w >= 100:
                margin = border_margin_px
                if y1 < margin or x1 < margin or (orig_h - y2) < margin or (orig_w - x2) < margin:
                    passed_gating = False
                    reasons.append("boundary_margin")

            bbox_area = max(1, (y2 - y1) * (x2 - x1))
            full_mask = best_det.get("mask")
            if full_mask is not None and full_mask.size > 0:
                m_patch = full_mask[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]
                mask_area = float(np.count_nonzero(m_patch))
                mask_ratio = mask_area / float(bbox_area)
                if mask_ratio < min_mask_area_ratio:
                    passed_gating = False
                    reasons.append(f"low_mask_ratio={mask_ratio:.2f}")

        score = (conf * 100.0) + min(100.0, lap_var / 5.0)

        y1_c, x1_c, y2_c, x2_c = best_det.get("bbox_cropped", [0, 0, orig_h, orig_w])
        native_crop = frame_bgr[y1_c:y2_c, x1_c:x2_c].copy() if (y2_c > y1_c and x2_c > x1_c) else best_det.get("cropped_bgr", frame_bgr)
        full_mask = best_det.get("mask")
        crop_mask = None
        if full_mask is not None and full_mask.shape[0] >= y2_c and full_mask.shape[1] >= x2_c:
            crop_mask = (full_mask[y1_c:y2_c, x1_c:x2_c].astype(np.uint8) * 255) if (y2_c > y1_c and x2_c > x1_c) else None

        all_candidates.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(float(f_idx / fps), 2),
            "confidence": conf,
            "laplacian_var": lap_var,
            "score": score,
            "passed_gating": passed_gating,
            "rejection_reasons": reasons,
            "native_crop": native_crop,
            "crop_mask": crop_mask,
            "bbox": [y1_c, x1_c, y2_c, x2_c],
        })

    cap.release()

    # Step 3: Select top 5 highest-scoring distinct frames across video with temporal spread
    valid_candidates = [c for c in all_candidates if c["passed_gating"]]
    valid_candidates.sort(key=lambda c: c["score"], reverse=True)

    fallback_candidates = [c for c in all_candidates if not c["passed_gating"]]
    fallback_candidates.sort(key=lambda c: c["score"], reverse=True)

    selected: List[Dict[str, Any]] = []

    def try_select(pool: List[Dict[str, Any]], min_sep: int, current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        res = list(current)
        for cand in pool:
            if len(res) >= top_k_frames:
                break
            if all(abs(cand["frame_idx"] - s["frame_idx"]) >= min_sep for s in res):
                res.append(cand)
        return res

    # Attempt to pick with separation 15, then 10, then 5, then 1 from valid candidates
    for sep in (15, 10, 5, 1):
        selected = try_select(valid_candidates, sep, selected)
        if len(selected) >= 5:
            break

    # Gracefully fall back to best available candidate frames if fewer than 5 passed gating
    if len(selected) < 5:
        log(f"Only {len(selected)} valid frames passed strict gating; gracefully falling back to best available candidate frames.")
        for sep in (15, 10, 5, 1):
            selected = try_select(fallback_candidates, sep, selected)
            if len(selected) >= 5:
                break

    # Sort selected frames chronologically
    selected.sort(key=lambda c: c["frame_idx"])
    detected_candidates = selected[:top_k_frames]

    log(f"Selected {len(detected_candidates)} frames for groove extraction: {[c['frame_idx'] for c in detected_candidates]}")

    # Step 4: Process the 5 frames through canonical alignment, 2x Lanczos-4, CLAHE grayscale, groove filter, threshold, and skeleton pruning
    frames_records = []
    param_lists: Dict[str, List[float]] = {k: [] for k in FEATURE_KEYS}

    for i, candidate in enumerate(detected_candidates):
        f_idx = candidate["frame_idx"]
        native_bgr = candidate["native_crop"]
        crop_mask = candidate["crop_mask"]
        conf = candidate["confidence"]
        lap_var = candidate.get("laplacian_var", 0.0)

        log(f"[Frame {i + 1}/5] Processing muzzle from frame #{f_idx} (native size {native_bgr.shape[1]}x{native_bgr.shape[0]})...")

        # 1. Native crop save
        p_crop = cache_dir / f"frame_{i}_01_native_crop.jpg"
        cv2.imwrite(str(p_crop), native_bgr)

        # 2. Canonical alignment & 2x Lanczos-4 target sizing (448x448)
        aligned_bgr, aligned_mask, align_meta = canonical_align_muzzle(
            native_bgr, mask=crop_mask, target_size=canonical_size
        )
        angle_deg = float(align_meta.get("rotation_deg", 0.0))
        alignment_method = align_meta.get("alignment_method", "mask_principal_axis")
        upscaled_bgr = aligned_bgr
        mask_up = aligned_mask

        # Grayscale & CLAHE local contrast normalization on canonical aligned crop
        gray_up = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_up, (0, 0), sigmaX=1.5, sigmaY=1.5)
        gray_unsharp = cv2.addWeighted(gray_up, 1.5, blurred, -0.5, 0)
        clahe_filter = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe_filter.apply(gray_unsharp)

        p_enh = cache_dir / f"frame_{i}_02_enhanced_ridge.jpg"
        cv2.imwrite(str(p_enh), enhanced_gray)

        # 3. Stage 3 & 4: Groove Response & Thresholding
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
        raw_skeleton = groove_res["groove_skeleton"]

        # Prune skeleton (Task 3)
        pruned_skeleton = prune_skeleton(
            raw_skeleton,
            min_spur_length=min_spur_length,
            min_component_size=min_component_size,
        )

        # 4. Stage 5: Line Coordinates & Minutiae Parameters Extraction
        graph_res = extract_skeleton_graph(
            pruned_skeleton,
            mask=groove_res.get("mask"),
            prune=False,  # Already pruned
        )
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
        cv2.imwrite(str(p_skel), pruned_skeleton)
        cv2.imwrite(str(p_over), overlay)

        # Assemble the 10 numerical minutiae parameters from the pruned skeleton
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

        minutiae = [
            {
                "x": float(n["pos"][0]),
                "y": float(n["pos"][1]),
                "type": n["type"],
                "orientation": float(n.get("orientation", 0.0)),
            }
            for n in graph_res["node_list"]
        ]

        for k in FEATURE_KEYS:
            param_lists[k].append(params[k])

        rel_base = f"/output/video_five_frames/{cache_dir.name}"
        # Backend-only spatial feature arrays are held by the server's LRU cache.
        matcher_mask = np.where(np.asarray(mask_up) > 0, 255, 0).astype(np.uint8) if mask_up is not None else np.full((canonical_size, canonical_size), 255, dtype=np.uint8)
        matcher_skeleton = np.where((pruned_skeleton > 0) & (matcher_mask > 0), 255, 0).astype(np.uint8)
        matcher_gray = enhanced_gray.copy()
        # Store the matcher arrays after stage extraction; they remain backend-only.
        matcher_mask = np.where(np.asarray(mask_up) > 0, 255, 0).astype(np.uint8) if mask_up is not None else np.full((448, 448), 255, dtype=np.uint8)
        matcher_skeleton = np.where((pruned_skeleton > 0) & (matcher_mask > 0), 255, 0).astype(np.uint8)
        matcher_gray = enhanced_gray.copy()
        frames_records.append({
            "frame_number": i + 1,
            "frame_idx": f_idx,
            "timestamp_sec": candidate["timestamp_sec"],
            "confidence": round(float(conf), 3),
            "laplacian_var": round(float(lap_var), 2),
            "canonical_angle_deg": round(float(angle_deg), 2),
            "crop_url": f"{rel_base}/frame_{i}_01_native_crop.jpg",
            "enhanced_url": f"{rel_base}/frame_{i}_02_enhanced_ridge.jpg",
            "groove_response_url": f"{rel_base}/frame_{i}_03_groove_response.png",
            "groove_mask_url": f"{rel_base}/frame_{i}_04_groove_mask.png",
            "skeleton_url": f"{rel_base}/frame_{i}_05_thin_skeleton.png",
            "overlay_url": f"{rel_base}/frame_{i}_overlay.png",
            "crop_path": str(p_crop),
            "enhanced_path": str(p_enh),
            "groove_response_path": str(p_resp),
            "groove_mask_path": str(p_mask),
            "skeleton_path": str(p_skel),
            "overlay_path": str(p_over),
            "parameters": params,
            "clahe_gray": matcher_gray,
            "mask": matcher_mask,
            "skeleton": matcher_skeleton,
            "alignment_method": alignment_method,
            "rotation_deg": round(float(angle_deg), 2),
            "minutiae": minutiae,
        })
        log(f" -> [Frame {i + 1}/5] Lines extracted: {params['skeleton_pixels']} skel px, {params['endpoints']} ends, {params['junctions']} juncs, {params['cycles']} loops, {len(minutiae)} minutiae")

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

    # Matcher-only NumPy arrays stay in memory for the API cache and are never
    # written to JSON files or exposed in HTTP JSON responses.
    summary_payload = dict(result)
    summary_payload["frames"] = [
        {key: value for key, value in frame.items() if not isinstance(value, np.ndarray)}
        for frame in frames_records
    ]
    with open(cache_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    log(f"Completed 5-frame groove parameter extraction for {v_path.name} successfully.")
    return result


extract_five_frames_groove_features = extract_five_muzzle_stages

