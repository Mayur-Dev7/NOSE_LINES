#!/usr/bin/env python3
"""
Five-Image Batch Processing and Quality Assessment Module
=========================================================
Performs automated multi-image quality assessment, exposure and sharpness analysis,
near-duplicate detection, and 5-stage morphological groove extraction for cattle muzzle
biometric reference template generation.

Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import cv2
import numpy as np

# Ensure parent and current directory are on sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from groove_extractor import extract_muzzle_grooves_pipeline, isolate_muzzle_region
from template_schema import (
    make_empty_template,
    robust_median_and_mad,
    calculate_stability,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

# Default Quality Assessment Thresholds
DEFAULT_QUALITY_THRESHOLDS: Dict[str, float] = {
    "min_sharpness": 30.0,             # Minimum Laplacian variance (CV_64F)
    "min_contrast": 12.0,              # Minimum standard deviation of grayscale intensities
    "max_underexposed_ratio": 0.30,    # Maximum ratio of clipped dark pixels (< 10)
    "max_overexposed_ratio": 0.30,     # Maximum ratio of clipped bright pixels (> 245)
    "min_mean_intensity": 20.0,        # Minimum mean intensity for cattle muzzle
    "max_mean_intensity": 235.0,       # Maximum mean intensity (above which is washed out)
    "min_area_pct": 15.0,              # Minimum valid muzzle mask area percentage
    "max_border_touches": 1,           # Max allowable border contacts before crop truncation (>= 2 is truncated)
    "border_contact_ratio": 0.02,      # Minimum border length contact ratio to register border touch
}


def assess_image_quality(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates measurable quality signals of a cattle muzzle image or crop.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR (H, W, 3) or Grayscale (H, W) format.
    mask : Optional[np.ndarray], default=None
        Binary mask (H, W) or (H, W, 3) representing the muzzle region. If None,
        the muzzle region is isolated automatically using adaptive morphological priors.
    thresholds : Optional[Dict[str, float]], default=None
        Custom dictionary overriding default quality thresholds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'sharpness': float, Laplacian variance
        - 'laplacian_variance': float, alias for sharpness
        - 'mean_intensity': float, mean grayscale intensity of muzzle ROI
        - 'underexposed_ratio': float, fraction of pixels < 10
        - 'overexposed_ratio': float, fraction of pixels > 245
        - 'exposure_clipping_ratio': float, combined clipping fraction
        - 'contrast': float, standard deviation of grayscale intensities
        - 'contrast_std': float, alias for contrast
        - 'muzzle_area_pixels': int, number of non-zero pixels in mask
        - 'muzzle_area_pct': float, area percentage relative to crop
        - 'touched_borders': List[str], borders contacted by mask
        - 'touched_border_count': int, count of touched borders
        - 'is_truncated': bool, True if mask contacts multiple borders
        - 'overall_quality_score': float, quality metric in [0.0, 100.0]
        - 'is_usable': bool, True if no rejection reasons triggered
        - 'rejection_reasons': List[str], reasons for rejection if unusable
    """
    cfg = DEFAULT_QUALITY_THRESHOLDS.copy()
    if thresholds:
        cfg.update(thresholds)

    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return {
            "sharpness": 0.0,
            "laplacian_variance": 0.0,
            "mean_intensity": 0.0,
            "underexposed_ratio": 1.0,
            "overexposed_ratio": 0.0,
            "exposure_clipping_ratio": 1.0,
            "contrast": 0.0,
            "contrast_std": 0.0,
            "muzzle_area_pixels": 0,
            "muzzle_area_pct": 0.0,
            "touched_borders": [],
            "touched_border_count": 0,
            "is_truncated": False,
            "overall_quality_score": 0.0,
            "is_usable": False,
            "rejection_reasons": ["empty_image"],
        }

    # Normalize image to BGR and Grayscale
    if image_bgr.ndim == 2:
        gray = image_bgr.copy()
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
        gray = image_bgr[:, :, 0].copy()
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3:
        img_bgr = image_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image_bgr.shape}")

    h, w = gray.shape[:2]
    total_pixels = max(1, h * w)

    # Resolve Mask
    if mask is not None:
        if mask.ndim == 3:
            m_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            m_gray = mask
        if m_gray.shape[:2] != (h, w):
            m_gray = cv2.resize(m_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        clean_mask = np.where(m_gray > 0, 255, 0).astype(np.uint8)
    else:
        # Automatically isolate muzzle ROI using project morphological engine
        clean_mask, _ = isolate_muzzle_region(img_bgr, mask=None, exclude_nostrils=True)

    # 1. Valid Muzzle Area
    muzzle_area_pixels = int(np.count_nonzero(clean_mask))
    muzzle_area_pct = round(float((muzzle_area_pixels / total_pixels) * 100.0), 2)

    # 2. Crop Completeness Check
    # A mask touching multiple borders (>= 2) indicates a truncated crop
    min_contact_h = max(2, int(w * cfg["border_contact_ratio"]))
    min_contact_v = max(2, int(h * cfg["border_contact_ratio"]))

    touched_borders: List[str] = []
    if np.count_nonzero(clean_mask[0, :]) >= min_contact_h:
        touched_borders.append("top")
    if np.count_nonzero(clean_mask[h - 1, :]) >= min_contact_h:
        touched_borders.append("bottom")
    if np.count_nonzero(clean_mask[:, 0]) >= min_contact_v:
        touched_borders.append("left")
    if np.count_nonzero(clean_mask[:, w - 1]) >= min_contact_v:
        touched_borders.append("right")

    touched_border_count = len(touched_borders)
    is_truncated = touched_border_count > int(cfg["max_border_touches"])

    # 3. Sharpness: Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = round(float(laplacian.var()), 2)

    # 4. Exposure and Contrast
    if muzzle_area_pixels > 0:
        roi_pixels = gray[clean_mask > 0]
    else:
        roi_pixels = gray.ravel()

    mean_intensity = round(float(np.mean(roi_pixels)), 2)
    contrast = round(float(np.std(roi_pixels)), 2)

    under_count = int(np.count_nonzero(roi_pixels < 10))
    over_count = int(np.count_nonzero(roi_pixels > 245))
    roi_size = max(1, roi_pixels.size)

    underexposed_ratio = round(float(under_count / roi_size), 4)
    overexposed_ratio = round(float(over_count / roi_size), 4)
    exposure_clipping_ratio = round(float(underexposed_ratio + overexposed_ratio), 4)

    # 5. Rejection Classification
    rejection_reasons: List[str] = []

    if sharpness < cfg["min_sharpness"]:
        rejection_reasons.append("severe_blur")

    if (
        underexposed_ratio > cfg["max_underexposed_ratio"]
        or mean_intensity < cfg["min_mean_intensity"]
    ):
        rejection_reasons.append("underexposed")

    if (
        overexposed_ratio > cfg["max_overexposed_ratio"]
        or mean_intensity > cfg["max_mean_intensity"]
    ):
        rejection_reasons.append("overexposed")

    if muzzle_area_pct < cfg["min_area_pct"]:
        rejection_reasons.append("insufficient_area")

    if is_truncated:
        rejection_reasons.append("truncated_crop")

    # Contrast check (only if not already underexposed or blurred to avoid redundant flags)
    if (
        contrast < cfg["min_contrast"]
        and "underexposed" not in rejection_reasons
        and "severe_blur" not in rejection_reasons
    ):
        rejection_reasons.append("low_contrast")

    is_usable = len(rejection_reasons) == 0

    # 6. Overall Quality Score (0.0 - 100.0)
    # Balanced formulation tailored for cattle/buffalo muzzle biometrics:
    # Sharpness normalized up to 120 (sharp muzzle features)
    s_score = float(np.clip(sharpness / 120.0, 0.0, 1.0) * 100.0)

    # Contrast normalized up to 35
    c_score = float(np.clip((contrast - 10.0) / 25.0, 0.0, 1.0) * 100.0)

    # Exposure score: cattle muzzles are dark pigmented (mean 30-160 is optimal)
    if 30.0 <= mean_intensity <= 160.0:
        mean_factor = 1.0
    elif mean_intensity < 30.0:
        mean_factor = max(0.0, (mean_intensity - 10.0) / 20.0)
    else:
        mean_factor = max(0.0, (240.0 - mean_intensity) / 80.0)

    clip_factor = max(0.0, 1.0 - 2.5 * exposure_clipping_ratio)
    e_score = float(np.clip(mean_factor * clip_factor * 100.0, 0.0, 100.0))

    # Area score: normalized relative to 50% target coverage
    a_score = float(np.clip(muzzle_area_pct / 50.0, 0.0, 1.0) * 100.0)

    # Completeness multiplier
    comp_factor = 0.5 if is_truncated else 1.0

    raw_score = (0.35 * s_score + 0.25 * c_score + 0.25 * e_score + 0.15 * a_score) * comp_factor

    if not is_usable:
        # Unusable crops are strictly capped below 50
        overall_quality_score = round(float(np.clip(min(raw_score, 48.0), 0.0, 48.0)), 2)
    else:
        overall_quality_score = round(float(np.clip(raw_score, 50.0, 100.0)), 2)

    return {
        "sharpness": sharpness,
        "laplacian_variance": sharpness,
        "mean_intensity": mean_intensity,
        "underexposed_ratio": underexposed_ratio,
        "overexposed_ratio": overexposed_ratio,
        "exposure_clipping_ratio": exposure_clipping_ratio,
        "contrast": contrast,
        "contrast_std": contrast,
        "muzzle_area_pixels": muzzle_area_pixels,
        "muzzle_area_pct": muzzle_area_pct,
        "touched_borders": touched_borders,
        "touched_border_count": touched_border_count,
        "is_truncated": is_truncated,
        "overall_quality_score": overall_quality_score,
        "is_usable": is_usable,
        "rejection_reasons": rejection_reasons,
    }


def detect_near_duplicates(
    images: List[np.ndarray],
    threshold: float = 0.92,
    method: str = "ncc",
    downsample_dim: Tuple[int, int] = (64, 64),
) -> List[Tuple[int, int, float]]:
    """
    Detects near-duplicate images within a batch using downsampled cross-correlation
    or structural similarity.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images in BGR or Grayscale format.
    threshold : float, default=0.92
        Similarity threshold above which two images are considered duplicates.
    method : str, default='ncc'
        Similarity metric: 'ncc' (Normalized Cross-Correlation) or 'ssim' (Structural Similarity).
    downsample_dim : Tuple[int, int], default=(64, 64)
        Grayscale downsampling resolution for fast pairwise comparisons.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of tuples `(i, j, similarity)` where `i < j` and `similarity >= threshold`.
    """
    if not images or len(images) < 2:
        return []

    # Preprocess each image to downsampled grayscale float32
    downsampled: List[np.ndarray] = []
    for img in images:
        if img is None or img.size == 0:
            downsampled.append(np.zeros(downsample_dim, dtype=np.float32))
            continue

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        resized = cv2.resize(gray, downsample_dim, interpolation=cv2.INTER_AREA)
        downsampled.append(resized)

    duplicates: List[Tuple[int, int, float]] = []
    n = len(images)

    for i in range(n):
        for j in range(i + 1, n):
            img_i = downsampled[i]
            img_j = downsampled[j]

            std_i = float(np.std(img_i))
            std_j = float(np.std(img_j))

            if std_i < 1e-4 and std_j < 1e-4:
                # Both images are uniform: check mean absolute difference
                mean_diff = abs(float(np.mean(img_i)) - float(np.mean(img_j)))
                sim = 1.0 if mean_diff < 5.0 else 0.0
            elif std_i < 1e-4 or std_j < 1e-4:
                # One uniform, one structured
                sim = 0.0
            elif method.lower() == "ssim":
                from skimage.metrics import structural_similarity as compute_ssim
                sim = float(compute_ssim(img_i.astype(np.uint8), img_j.astype(np.uint8), data_range=255))
                sim = max(0.0, min(1.0, sim))
            else:
                # Standard Normalized Cross-Correlation (cv2.TM_CCOEFF_NORMED)
                res = cv2.matchTemplate(
                    img_i.astype(np.float32),
                    img_j.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                sim = float(res[0, 0])
                sim = max(0.0, min(1.0, sim))

            if sim >= threshold:
                duplicates.append((i, j, round(float(sim), 4)))

    return sorted(duplicates, key=lambda x: (x[0], x[1]))


def _to_json_serializable(obj: Any) -> Any:
    """Recursively converts numpy data types to Python built-in types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def process_image_batch(
    images: List[Union[str, Path, np.ndarray]],
    output_dir: Optional[Union[str, Path]] = None,
    groove_params: Optional[Dict[str, Any]] = None,
    masks: Optional[List[Optional[np.ndarray]]] = None,
    duplicate_threshold: float = 0.92,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingests an image batch (up to 5 images), assesses quality, detects near-duplicates,
    and executes the 5-stage muzzle groove extraction pipeline.

    Parameters
    ----------
    images : List[Union[str, Path, np.ndarray]]
        List of input muzzle images (file paths or numpy arrays). Supports 1 to 5 images.
    output_dir : Optional[Union[str, Path]], default=None
        Directory to save intermediate and final artifacts.
    groove_params : Optional[Dict[str, Any]], default=None
        Parameters passed to extract_muzzle_grooves_pipeline (method, scales, threshold_method, etc.).
    masks : Optional[List[Optional[np.ndarray]]], default=None
        Optional pre-computed muzzle segmentation masks matching the images list.
    duplicate_threshold : float, default=0.92
        Near-duplicate correlation threshold.
    batch_id : Optional[str], default=None
        Identifier for the batch template.

    Returns
    -------
    Dict[str, Any]
        Structured dictionary containing:
        - 'records': list of per-image records
        - 'usable_count': int
        - 'submitted_count': int
        - 'duplicate_count': int
        - 'independent_view_count': int
        - 'rejected_count': int
        - 'near_duplicates': List[Tuple[int, int, float]]
        - 'quality_summary': Dict[str, Any]
        - 'batch_summary': Dict[str, Any]
        - 'canonical_template': Dict[str, Any]
    """
    if batch_id is None:
        batch_id = f"batch_{int(time.time() * 1000)}"

    submitted_count = len(images)
    records: List[Dict[str, Any]] = []
    ingested_images: List[np.ndarray] = []

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    g_params = {
        "method": "blackhat",
        "scales": [9, 15, 21],
        "threshold_method": "otsu",
        "min_component_size": 15,
        "exclude_nostrils": True,
    }
    if groove_params:
        g_params.update(groove_params)

    # Step 1: Ingest and validate each image
    for idx, item in enumerate(images):
        source_id = f"img_{idx:02d}"
        img_bgr: Optional[np.ndarray] = None

        if isinstance(item, (str, Path)):
            p = Path(item)
            source_id = p.stem
            if p.is_file():
                img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                logger.warning(f"Could not read image from path: {item}")
        elif isinstance(item, np.ndarray):
            source_id = f"image_{idx:02d}"
            if item.ndim == 2:
                img_bgr = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
            elif item.ndim == 3 and item.shape[2] == 1:
                img_bgr = cv2.cvtColor(item[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif item.ndim == 3:
                img_bgr = item.copy()
        elif isinstance(item, tuple) and len(item) >= 1:
            # Handle tuple (image, mask) if passed
            sub_img = item[0]
            if isinstance(sub_img, np.ndarray):
                img_bgr = sub_img.copy() if sub_img.ndim == 3 else cv2.cvtColor(sub_img, cv2.COLOR_GRAY2BGR)

        if img_bgr is None:
            # Create a dummy image record for failed load
            dummy_quality = assess_image_quality(np.zeros((10, 10, 3), dtype=np.uint8))
            dummy_quality["rejection_reasons"] = ["load_failed"]
            dummy_quality["is_usable"] = False
            records.append({
                "image_index": idx,
                "source_id": source_id,
                "dimensions": (0, 0),
                "quality": dummy_quality,
                "is_usable": False,
                "is_duplicate": False,
                "rejection_reasons": ["load_failed"],
                "stage_outputs": {},
                "clean_mask": np.zeros((0, 0), dtype=np.uint8),
                "masked_image": np.zeros((0, 0, 3), dtype=np.uint8),
                "enhanced_response": np.zeros((0, 0), dtype=np.uint8),
                "groove_mask": np.zeros((0, 0), dtype=np.uint8),
                "groove_skeleton": np.zeros((0, 0), dtype=np.uint8),
                "groove_overlay": np.zeros((0, 0, 3), dtype=np.uint8),
                "saved_paths": {},
                "timing_ms": 0.0,
                "groove_pixel_count": 0,
                "skeleton_pixel_count": 0,
                "groove_coverage_pct": 0.0,
            })
            ingested_images.append(np.zeros((64, 64, 3), dtype=np.uint8))
            continue

        h, w = img_bgr.shape[:2]
        ingested_images.append(img_bgr)

        # Retrieve optional mask
        img_mask: Optional[np.ndarray] = None
        if masks and idx < len(masks):
            img_mask = masks[idx]

        # Step 2: Quality Assessment
        quality = assess_image_quality(img_bgr, mask=img_mask)

        # Step 3: Run 5-stage groove extraction pipeline
        pipeline_prefix = f"img_{idx:02d}"
        pipeline_out_dir = str(output_dir) if output_dir else None

        pipeline_res = extract_muzzle_grooves_pipeline(
            image_bgr=img_bgr,
            mask=img_mask,
            method=g_params["method"],
            scales=g_params["scales"],
            threshold_method=g_params["threshold_method"],
            min_component_size=g_params["min_component_size"],
            exclude_nostrils=g_params["exclude_nostrils"],
            output_dir=pipeline_out_dir,
            prefix=pipeline_prefix,
        )

        stage_outputs = {
            "clean_mask": pipeline_res["clean_mask"],
            "masked_image": pipeline_res["masked_image"],
            "enhanced_response": pipeline_res["enhanced_response"],
            "groove_mask": pipeline_res["groove_mask"],
            "groove_skeleton": pipeline_res["groove_skeleton"],
            "groove_overlay": pipeline_res["groove_overlay"],
        }

        record = {
            "image_index": idx,
            "source_id": source_id,
            "dimensions": (w, h),
            "quality": quality,
            "is_usable": quality["is_usable"],
            "is_duplicate": False,  # Updated after near-duplicate detection
            "rejection_reasons": list(quality["rejection_reasons"]),
            "stage_outputs": stage_outputs,
            "clean_mask": pipeline_res["clean_mask"],
            "masked_image": pipeline_res["masked_image"],
            "enhanced_response": pipeline_res["enhanced_response"],
            "groove_mask": pipeline_res["groove_mask"],
            "groove_skeleton": pipeline_res["groove_skeleton"],
            "groove_overlay": pipeline_res["groove_overlay"],
            "saved_paths": pipeline_res.get("saved_paths", {}),
            "timing_ms": pipeline_res.get("timing_ms", 0.0),
            "groove_pixel_count": pipeline_res.get("groove_pixel_count", 0),
            "skeleton_pixel_count": pipeline_res.get("skeleton_pixel_count", 0),
            "groove_coverage_pct": pipeline_res.get("groove_coverage_pct", 0.0),
        }
        records.append(record)

    # Step 4: Near-duplicate detection
    near_duplicates = detect_near_duplicates(
        ingested_images,
        threshold=duplicate_threshold,
        method="ncc",
    )

    # Mark duplicate records:
    # For every detected pair (i, j, sim), resolve representative
    duplicate_indices: set = set()
    for i, j, sim in near_duplicates:
        q_i = records[i]["is_usable"]
        q_j = records[j]["is_usable"]

        if q_i and not q_j:
            duplicate_indices.add(j)
        elif not q_i and q_j:
            duplicate_indices.add(i)
        else:
            # Both usable or both unusable: keep canonical i, mark j as duplicate
            duplicate_indices.add(j)

    for idx, rec in enumerate(records):
        if idx in duplicate_indices:
            rec["is_duplicate"] = True

    # Step 5: Compute Batch Aggregates
    processed_count = len(records)
    usable_count = sum(1 for r in records if r["is_usable"])
    duplicate_count = sum(1 for r in records if r["is_duplicate"])
    independent_view_count = sum(1 for r in records if r["is_usable"] and not r["is_duplicate"])
    rejected_count = sum(1 for r in records if not r["is_usable"])

    quality_scores = [r["quality"]["overall_quality_score"] for r in records]
    sharpness_values = [r["quality"]["sharpness"] for r in records]

    mean_quality = round(float(np.mean(quality_scores)), 2) if quality_scores else 0.0
    med_sharpness, _ = robust_median_and_mad(sharpness_values) if sharpness_values else (0.0, 0.0)
    usable_ratio = round(float(usable_count / max(1, processed_count)), 3)

    # Step 6: Assemble Canonical Reference Template
    canonical_template = make_empty_template(template_id=batch_id, submitted_count=submitted_count)
    canonical_template["source_images"] = [r["source_id"] for r in records]
    canonical_template["batch_summary"] = {
        "submitted_count": submitted_count,
        "processed_count": processed_count,
        "usable_count": usable_count,
        "independent_view_count": independent_view_count,
        "rejected_count": rejected_count,
    }
    canonical_template["quality_summary"] = {
        "mean_quality_score": mean_quality,
        "median_sharpness": round(float(med_sharpness), 2),
        "usable_ratio": usable_ratio,
        "duplicate_count": duplicate_count,
    }

    # Populate per-image feature summaries in canonical template
    for r in records:
        canonical_template["per_image_features"].append({
            "image_index": r["image_index"],
            "source_id": r["source_id"],
            "dimensions": r["dimensions"],
            "is_usable": r["is_usable"],
            "is_duplicate": r["is_duplicate"],
            "overall_quality_score": r["quality"]["overall_quality_score"],
            "sharpness": r["quality"]["sharpness"],
            "mean_intensity": r["quality"]["mean_intensity"],
            "contrast": r["quality"]["contrast"],
            "groove_pixel_count": r.get("groove_pixel_count", 0),
            "skeleton_pixel_count": r.get("skeleton_pixel_count", 0),
            "groove_coverage_pct": r.get("groove_coverage_pct", 0.0),
            "rejection_reasons": r["rejection_reasons"],
        })

    # Feature stability computation on usable records
    usable_records = [r for r in records if r["is_usable"] and not r["is_duplicate"]]
    if len(usable_records) >= 2:
        groove_counts = [float(r.get("groove_pixel_count", 0)) for r in usable_records]
        sharpnesses = [float(r["quality"]["sharpness"]) for r in usable_records]
        stab_grooves = calculate_stability(groove_counts)
        stab_sharp = calculate_stability(sharpnesses)
        overall_stab = round(float(0.6 * stab_grooves + 0.4 * stab_sharp), 3)
        confidence = "high" if overall_stab >= 0.8 else "moderate" if overall_stab >= 0.5 else "low"
        canonical_template["feature_stability"] = {
            "overall_stability_score": overall_stab,
            "groove_stability": stab_grooves,
            "sharpness_stability": stab_sharp,
            "confidence_level": confidence,
        }
    elif len(usable_records) == 1:
        canonical_template["feature_stability"] = {
            "overall_stability_score": 1.0,
            "confidence_level": "single_view",
        }

    canonical_template["processing_config"] = {
        "groove_params": g_params,
        "duplicate_threshold": duplicate_threshold,
    }

    # Save summary and canonical template JSON if output directory is provided
    if output_dir is not None:
        p_summary = os.path.join(output_dir, "batch_summary.json")
        p_template = os.path.join(output_dir, "canonical_template.json")

        summary_export = {
            "batch_id": batch_id,
            "schema_version": SCHEMA_VERSION,
            "submitted_count": submitted_count,
            "processed_count": processed_count,
            "usable_count": usable_count,
            "duplicate_count": duplicate_count,
            "independent_view_count": independent_view_count,
            "rejected_count": rejected_count,
            "quality_summary": canonical_template["quality_summary"],
            "batch_summary": canonical_template["batch_summary"],
            "near_duplicates": near_duplicates,
            "records": [
                {
                    "image_index": r["image_index"],
                    "source_id": r["source_id"],
                    "dimensions": r["dimensions"],
                    "is_usable": r["is_usable"],
                    "is_duplicate": r["is_duplicate"],
                    "rejection_reasons": r["rejection_reasons"],
                    "quality": r["quality"],
                    "saved_paths": r["saved_paths"],
                    "groove_pixel_count": r.get("groove_pixel_count", 0),
                    "skeleton_pixel_count": r.get("skeleton_pixel_count", 0),
                    "groove_coverage_pct": r.get("groove_coverage_pct", 0.0),
                }
                for r in records
            ],
        }

        try:
            with open(p_summary, "w") as f:
                json.dump(_to_json_serializable(summary_export), f, indent=2)
            with open(p_template, "w") as f:
                json.dump(_to_json_serializable(canonical_template), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save batch JSON files: {e}")

    return {
        "records": records,
        "usable_count": usable_count,
        "submitted_count": submitted_count,
        "duplicate_count": duplicate_count,
        "independent_view_count": independent_view_count,
        "rejected_count": rejected_count,
        "processed_count": processed_count,
        "near_duplicates": near_duplicates,
        "quality_summary": canonical_template["quality_summary"],
        "batch_summary": canonical_template["batch_summary"],
        "canonical_template": canonical_template,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Five-Image Batch Processor and Quality Assessor")
    parser.add_argument("--images", nargs="+", required=True, help="List of input image paths (up to 5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch artifacts")
    parser.add_argument("--dup-threshold", type=float, default=0.92, help="Duplicate similarity threshold")
    parser.add_argument("--method", type=str, default="blackhat", choices=["blackhat", "frangi"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    res = process_image_batch(
        images=args.images,
        output_dir=args.output_dir,
        groove_params={"method": args.method},
        duplicate_threshold=args.dup_threshold,
    )

    print(f"Batch Processing Completed:")
    print(f"  Submitted:   {res['submitted_count']}")
    print(f"  Usable:      {res['usable_count']}")
    print(f"  Duplicates:  {res['duplicate_count']}")
    print(f"  Independent: {res['independent_view_count']}")
    print(f"  Rejected:    {res['rejected_count']}")
    for r in res["records"]:
        status = "USABLE" if r["is_usable"] and not r["is_duplicate"] else "DUPLICATE" if r["is_duplicate"] else "REJECTED"
        print(f"  - [{r['image_index']}] {r['source_id']}: {status} (score={r['quality']['overall_quality_score']}, reasons={r['rejection_reasons']})")


if __name__ == "__main__":
    main()
