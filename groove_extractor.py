#!/usr/bin/env python3
"""
Muzzle Groove Extraction Engine
===============================
Extracts dermatoglyphic groove patterns, ridge networks, and bead boundary
skeletons from cattle/buffalo muzzle imagery using mathematical morphology
and ridge filtering (Black-Hat and Frangi).

Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import skimage.filters
import skimage.morphology

logger = logging.getLogger(__name__)


def isolate_muzzle_region(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    exclude_nostrils: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolates the cattle muzzle ROI and suppresses background / nostril cavities.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR format (or 2D grayscale), shape (H, W, 3) or (H, W).
    mask : Optional[np.ndarray], default=None
        Pre-computed binary mask from segmentation model. If None, creates an
        adaptive muzzle skin ROI mask based on skin color/texture and elliptical center prior.
    exclude_nostrils : bool, default=True
        Whether to detect and exclude dark nostril cavities in the upper muzzle region.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        clean_mask : uint8 ndarray of shape (H, W) with values 0 or 255.
        masked_image_bgr : uint8 ndarray of shape (H, W, 3) with background zeroed out.
    """
    img = image_bgr.copy()
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img
    else:
        img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    # Adaptive structuring element size based on resolution
    scale_factor = min(h, w) / 448.0
    k_close = max(3, int(round(9 * scale_factor)))
    if k_close % 2 == 0:
        k_close += 1
    k_open = max(3, int(round(5 * scale_factor)))
    if k_open % 2 == 0:
        k_open += 1

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    if mask is not None:
        if mask.ndim == 3:
            mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_2d = mask
        if mask_2d.shape[:2] != (h, w):
            mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
        clean_mask = np.where(mask_2d > 0, 255, 0).astype(np.uint8)

        if np.count_nonzero(clean_mask) > 0:
            # Morphologically close to fill holes
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
            # Morphologically open to prune spurs
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
    else:
        # Elliptical center prior covering central 90% of width and 84% of height
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.42))
        prior_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(prior_mask, center, axes, 0, 0, 360, 255, -1)

        std_val = float(np.std(gray))
        if std_val < 5.0:
            # Uniform or near-uniform synthetic image
            clean_mask = prior_mask
        else:
            # Adaptive skin ROI based on center skin sample and elliptical prior
            y1, y2 = int(h * 0.35), int(h * 0.65)
            x1, x2 = int(w * 0.35), int(w * 0.65)
            center_sample = gray[y1:y2, x1:x2]
            c_mean = float(np.mean(center_sample))
            c_std = max(10.0, float(np.std(center_sample)))

            # Exclude extreme background outliers (> 3.5 std from center skin)
            diff = np.abs(gray.astype(np.float32) - c_mean)
            skin_candidates = (diff < 3.5 * c_std).astype(np.uint8) * 255
            clean_mask = cv2.bitwise_and(skin_candidates, prior_mask)

            # Morphological cleaning
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
            # Ensure center core is always solid
            clean_mask[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)] = 255

    # Nostril Cavity Exclusion
    if exclude_nostrils and np.count_nonzero(clean_mask) > 0:
        skin_pixels = gray[clean_mask > 0]
        if len(skin_pixels) > 0:
            s_mean = float(np.mean(skin_pixels))
            s_std = float(np.std(skin_pixels))

            # Nostril cavities are very dark cavities situated in the upper muzzle region
            if s_std >= 5.0:
                dark_thresh = min(60.0, s_mean - 1.2 * s_std)
                if dark_thresh > 15.0:
                    upper_limit = int(h * 0.60)
                    dark_upper = np.zeros_like(clean_mask)
                    dark_upper[:upper_limit, :] = (
                        (gray[:upper_limit, :] < dark_thresh)
                        & (clean_mask[:upper_limit, :] > 0)
                    ).astype(np.uint8)

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        dark_upper, connectivity=8
                    )
                    min_nostril_area = 0.005 * (h * w)
                    max_nostril_area = 0.35 * (h * w)

                    nostril_mask = np.zeros_like(clean_mask)
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        cw = stats[i, cv2.CC_STAT_WIDTH]
                        ch = stats[i, cv2.CC_STAT_HEIGHT]
                        aspect = max(cw, ch) / max(1, min(cw, ch))
                        solidity = area / max(1, cw * ch)
                        cy = centroids[i][1]

                        # Nostrils are large, solid, moderately rounded dark cavities in upper region
                        if (
                            min_nostril_area <= area <= max_nostril_area
                            and aspect <= 3.5
                            and solidity >= 0.30
                            and cy < h * 0.55
                        ):
                            nostril_mask[labels == i] = 255

                    if np.count_nonzero(nostril_mask) > 0:
                        # Slightly dilate nostril mask to cleanly seal cavity edges
                        k_nostril = max(3, int(round(7 * scale_factor)))
                        if k_nostril % 2 == 0:
                            k_nostril += 1
                        nostril_mask = cv2.dilate(
                            nostril_mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_nostril, k_nostril)),
                        )
                        clean_mask[nostril_mask > 0] = 0

    clean_mask = (clean_mask > 0).astype(np.uint8) * 255
    masked_image_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)

    return clean_mask, masked_image_bgr


def enhance_dark_grooves(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    method: str = "blackhat",
    scales: List[int] = [9, 15, 21],
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhances dark valleys (grooves/ridges) using multi-scale Black-Hat morphology or Frangi filter.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR or Grayscale format.
    mask : np.ndarray
        Muzzle ROI mask (uint8 0 or 255).
    method : str, default='blackhat'
        Enhancement algorithm: 'blackhat' or 'frangi'.
    scales : List[int], default=[9, 15, 21]
        Base kernel scales for multi-scale Black-Hat morphology at 448x448 standard.
    use_clahe : bool, default=True
        Whether to apply Contrast Limited Adaptive Histogram Equalization.
    clahe_clip_limit : float, default=2.0
        CLAHE clip limit.
    clahe_grid_size : Tuple[int, int], default=(8, 8)
        CLAHE grid tile size.

    Returns
    -------
    np.ndarray
        enhanced_response : uint8 ndarray [0, 255] with grooves brightened and background 0.
    """
    img = image_bgr.copy()
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]

    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Return zeros immediately if mask is empty
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    # Apply CLAHE if enabled
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size,
        )
        gray = clahe.apply(gray)

    # Resolution scaling relative to standard 448x448
    scale_factor = min(h, w) / 448.0

    method_lower = method.lower()
    if method_lower == "blackhat":
        responses = []
        for s in scales:
            k = int(round(s * scale_factor))
            k = max(3, k)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            responses.append(bh)

        if not responses:
            return np.zeros_like(gray, dtype=np.uint8)

        combined = np.maximum.reduce(responses)

        # Normalize to [0, 255]
        max_val = float(combined.max())
        min_val = float(combined.min())
        if max_val > min_val and max_val > 0:
            enhanced = ((combined.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).astype(
                np.uint8
            )
        else:
            enhanced = np.zeros_like(gray, dtype=np.uint8)

    elif method_lower == "frangi":
        gray_float = gray.astype(np.float64) / 255.0
        base_sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
        sigmas = [max(0.5, s * scale_factor) for s in base_sigmas]
        sigmas = sorted(list(set(sigmas)))

        frangi_resp = skimage.filters.frangi(
            gray_float,
            sigmas=sigmas,
            black_ridges=True,
        )

        f_max = float(frangi_resp.max())
        f_min = float(frangi_resp.min())
        if f_max > f_min and f_max > 0:
            enhanced = ((frangi_resp - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
        else:
            enhanced = np.zeros_like(gray, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown enhancement method '{method}'. Choose 'blackhat' or 'frangi'.")

    # Strictly zero out background outside mask
    enhanced[mask == 0] = 0
    return enhanced


def binarize_grooves(
    enhanced_response: np.ndarray,
    mask: np.ndarray,
    threshold_method: str = "otsu",
    adaptive_block_size: int = 15,
    adaptive_c: int = 2,
    min_component_size: int = 15,
) -> np.ndarray:
    """
    Binarizes groove responses using Otsu or Adaptive Gaussian thresholding,
    filtering out spurious noise blobs while preserving elongated groove lines.

    Parameters
    ----------
    enhanced_response : np.ndarray
        uint8 groove response from enhance_dark_grooves.
    mask : np.ndarray
        uint8 muzzle ROI mask (0 or 255).
    threshold_method : str, default='otsu'
        'otsu' or 'adaptive'.
    adaptive_block_size : int, default=15
        Block size for adaptive thresholding.
    adaptive_c : int, default=2
        Constant subtracted from mean in adaptive thresholding.
    min_component_size : int, default=15
        Minimum pixel area threshold for connected component noise pruning.

    Returns
    -------
    np.ndarray
        binary_groove_mask : uint8 ndarray (0 or 255) containing segmented grooves.
    """
    h, w = enhanced_response.shape[:2]
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Guard against empty response or mask
    if (
        mask is None
        or np.count_nonzero(mask) == 0
        or enhanced_response is None
        or enhanced_response.max() == 0
    ):
        return np.zeros_like(enhanced_response, dtype=np.uint8)

    method_lower = threshold_method.lower()
    binary = np.zeros_like(enhanced_response, dtype=np.uint8)

    if method_lower == "otsu":
        non_zero_masked = enhanced_response[(mask > 0) & (enhanced_response > 0)]
        if len(non_zero_masked) == 0:
            return binary

        if non_zero_masked.max() == non_zero_masked.min():
            thresh_val = float(non_zero_masked.min())
        else:
            thresh_val, _ = cv2.threshold(
                non_zero_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        binary[(enhanced_response >= max(1.0, thresh_val)) & (mask > 0)] = 255

    elif method_lower == "adaptive":
        bs = adaptive_block_size
        if bs % 2 == 0:
            bs += 1
        min_dim = min(h, w)
        if bs >= min_dim:
            bs = max(3, (min_dim - 1) | 1)

        raw_ad = cv2.adaptiveThreshold(
            enhanced_response,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            adaptive_c,
        )
        # Only retain positive groove responses inside mask
        binary[(raw_ad > 0) & (enhanced_response > 0) & (mask > 0)] = 255
    else:
        raise ValueError(
            f"Unknown threshold_method '{threshold_method}'. Choose 'otsu' or 'adaptive'."
        )

    # Strictly suppress outside mask
    binary[mask == 0] = 0

    if np.count_nonzero(binary) == 0:
        return binary

    # Connected Component Filtering: Prune noise, retain elongated lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = binary.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(cw, ch) / max(1, min(cw, ch))

        if area < min_component_size:
            comp_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            perimeter_to_area = 0.0
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                perimeter_to_area = perimeter / max(1.0, float(area))

            # Retain if elongated (aspect ratio > 3.0 or high perimeter/area ratio)
            if aspect_ratio > 3.0 or perimeter_to_area > 1.2:
                continue
            filtered[labels == i] = 0

    return filtered


def skeletonize_grooves(binary_mask: np.ndarray) -> np.ndarray:
    """
    Thins the binary groove mask to 1-pixel-wide topological centerlines.

    Parameters
    ----------
    binary_mask : np.ndarray
        uint8 binary mask with values 0 or 255.

    Returns
    -------
    np.ndarray
        groove_skeleton : uint8 ndarray (0 or 255) representing 1-pixel-wide centerlines.
    """
    if binary_mask is None or binary_mask.size == 0 or np.count_nonzero(binary_mask) == 0:
        return (
            np.zeros_like(binary_mask, dtype=np.uint8)
            if binary_mask is not None
            else np.zeros((0, 0), dtype=np.uint8)
        )

    bool_mask = binary_mask > 0
    skel = skimage.morphology.skeletonize(bool_mask)
    return (skel.astype(np.uint8)) * 255


def create_groove_overlay(
    image_bgr: np.ndarray,
    groove_skeleton: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlays colored groove skeleton centerlines on top of the original image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original image in BGR or Grayscale format.
    groove_skeleton : np.ndarray
        uint8 skeleton mask (0 or 255).
    color : Tuple[int, int, int], default=(0, 255, 0)
        BGR color for skeleton overlay (default bright green).

    Returns
    -------
    np.ndarray
        overlay : BGR image with colored groove centerlines.
    """
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    if groove_skeleton is not None and groove_skeleton.size > 0:
        overlay[groove_skeleton > 0] = color

    return overlay


def extract_muzzle_grooves_pipeline(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "blackhat",
    scales: List[int] = [9, 15, 21],
    threshold_method: str = "otsu",
    adaptive_block_size: int = 15,
    adaptive_c: int = 2,
    min_component_size: int = 15,
    exclude_nostrils: bool = True,
    output_dir: Optional[str] = None,
    prefix: str = "groove",
) -> Dict[str, Any]:
    """
    End-to-end pipeline extracting dermatoglyphic grooves, binary mask, and skeleton.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input muzzle image in BGR format (guaranteed NEVER modified in-place).
    mask : Optional[np.ndarray], default=None
        Optional muzzle mask.
    method : str, default='blackhat'
        Enhancement method: 'blackhat' or 'frangi'.
    scales : List[int], default=[9, 15, 21]
        Base kernel scales for multi-scale Black-Hat morphology.
    threshold_method : str, default='otsu'
        'otsu' or 'adaptive'.
    min_component_size : int, default=15
        Minimum component area for noise filtering.
    exclude_nostrils : bool, default=True
        Whether to exclude dark nostril cavities in upper muzzle region.
    output_dir : Optional[str], default=None
        Directory to save intermediate and final image artifacts.
    prefix : str, default='groove'
        Filename prefix for saved outputs.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing timing, resolution, arrays, and saved file paths.
    """
    t_start = time.perf_counter()
    img_copy = image_bgr.copy()
    h, w = img_copy.shape[:2]

    # Stage 1: Muzzle ROI isolation & nostril cavity suppression
    clean_mask, masked_img = isolate_muzzle_region(
        img_copy,
        mask=mask,
        exclude_nostrils=exclude_nostrils,
    )

    # Stage 2: Groove enhancement (multi-scale Black-Hat / Frangi)
    enhanced = enhance_dark_grooves(
        masked_img,
        clean_mask,
        method=method,
        scales=scales,
    )

    # Stage 3: Binarization and connected component filtering
    groove_mask = binarize_grooves(
        enhanced,
        clean_mask,
        threshold_method=threshold_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        min_component_size=min_component_size,
    )

    # Stage 4: Topological skeletonization
    groove_skeleton = skeletonize_grooves(groove_mask)

    # Stage 5: Visual overlay
    groove_overlay = create_groove_overlay(img_copy, groove_skeleton)

    # File export if output directory provided
    saved_paths: Dict[str, str] = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        p_resp = os.path.join(output_dir, f"{prefix}_response.png")
        p_mask = os.path.join(output_dir, f"{prefix}_mask.png")
        p_skel = os.path.join(output_dir, f"{prefix}_skeleton.png")
        p_over = os.path.join(output_dir, f"{prefix}_overlay.png")

        cv2.imwrite(p_resp, enhanced)
        cv2.imwrite(p_mask, groove_mask)
        cv2.imwrite(p_skel, groove_skeleton)
        cv2.imwrite(p_over, groove_overlay)

        saved_paths["groove_response"] = p_resp
        saved_paths["groove_mask"] = p_mask
        saved_paths["groove_skeleton"] = p_skel
        saved_paths["groove_overlay"] = p_over
        saved_paths["response"] = p_resp
        saved_paths["mask"] = p_mask
        saved_paths["skeleton"] = p_skel
        saved_paths["overlay"] = p_over

    t_end = time.perf_counter()
    timing_ms = (t_end - t_start) * 1000.0

    num_grooves = int(np.count_nonzero(groove_mask))
    num_skel = int(np.count_nonzero(groove_skeleton))
    mask_pixels = max(1, int(np.count_nonzero(clean_mask)))
    groove_cov = round((num_grooves / mask_pixels) * 100.0, 2)

    return {
        "timing_ms": timing_ms,
        "resolution": (w, h),
        "image_shape": img_copy.shape,
        "clean_mask": clean_mask,
        "masked_image": masked_img,
        "enhanced_response": enhanced,
        "groove_mask": groove_mask,
        "groove_skeleton": groove_skeleton,
        "groove_overlay": groove_overlay,
        "saved_paths": saved_paths,
        "num_groove_pixels": num_grooves,
        "num_skeleton_pixels": num_skel,
        "groove_pixel_count": num_grooves,
        "skeleton_pixel_count": num_skel,
        "groove_coverage_pct": groove_cov,
    }


def main():
    parser = argparse.ArgumentParser(description="Muzzle Groove Extraction Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to input muzzle image")
    parser.add_argument("--mask", type=str, default=None, help="Path to binary muzzle mask (optional)")
    parser.add_argument("--method", type=str, default="blackhat", choices=["blackhat", "frangi"])
    parser.add_argument("--threshold", type=str, default="otsu", choices=["otsu", "adaptive"])
    parser.add_argument("--output-dir", type=str, default="./groove_output")
    parser.add_argument("--prefix", type=str, default="groove")
    parser.add_argument("--no-nostril-exclusion", action="store_true", help="Disable nostril exclusion")

    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: Could not read image at '{args.image}'", file=sys.stderr)
        sys.exit(1)

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) if args.mask else None

    result = extract_muzzle_grooves_pipeline(
        image_bgr=image_bgr,
        mask=mask,
        method=args.method,
        threshold_method=args.threshold,
        exclude_nostrils=not args.no_nostril_exclusion,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"Processed in {result['timing_ms']:.2f} ms")
    print(f"Resolution: {result['resolution']}")
    print(f"Groove pixels: {result['num_groove_pixels']}")
    print(f"Skeleton pixels: {result['num_skeleton_pixels']}")
    for k, v in result["saved_paths"].items():
        if k in ["groove_response", "groove_mask", "groove_skeleton", "groove_overlay"]:
            print(f"Saved {k}: {v}")


if __name__ == "__main__":
    main()
