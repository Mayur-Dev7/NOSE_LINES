#!/usr/bin/env python3
"""
cv_workers.py
=============
Standalone CPU multiprocessing worker module for Steps 3-5 of the Bovine Biometric pipeline:
- Step 3: Canonical Alignment & CLAHE local contrast normalization
- Step 4: Multi-scale Frangi Hessian / Black-Hat dermatoglyphic groove extraction
- Step 5: Thin skeletonization, graph topology, and minutiae extraction

Strict constraints:
- Imports ONLY numpy, cv2, skimage (and standard library / groove_extractor / groove_graph).
- NEVER imports server.py, TensorFlow, or PyTorch.
- Worker initializer sets single-thread affinity (cv2.setNumThreads(1), OMP_NUM_THREADS=1, etc.).
- Persistent spawn process pool pre-forked before TF/CUDA initialization.
- Guarded with `if __name__ == '__main__':`.
"""

import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing as mp

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import skimage.filters
import skimage.morphology

from groove_extractor import extract_muzzle_grooves_pipeline
from groove_graph import (
    prune_skeleton,
    extract_skeleton_graph,
    visualize_groove_graph,
    extract_minutiae_points,
)


def worker_init():
    """Initializes worker process with strictly single-threaded math libraries."""
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_physical_cores() -> int:
    """Returns detected physical CPU cores count."""
    try:
        import glob
        cores = set()
        for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
            try:
                with open(p) as f:
                    cores.add(f.read().strip())
            except Exception:
                pass
        if cores:
            return len(cores)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


_GLOBAL_POOL: Optional[Any] = None
_CURRENT_POOL_SIZE: Optional[int] = None


def get_worker_pool(pool_size: Optional[int] = None) -> Any:
    """
    Returns or lazily creates the persistent spawn process pool.
    Pool size defaults to 2 for CPU-2 profiles, min(physical_cores, 6) otherwise.
    """
    global _GLOBAL_POOL, _CURRENT_POOL_SIZE
    if pool_size is None:
        pool_size = min(get_physical_cores(), 6)

    if _GLOBAL_POOL is not None and _CURRENT_POOL_SIZE != pool_size:
        try:
            _GLOBAL_POOL.close()
            _GLOBAL_POOL.terminate()
            _GLOBAL_POOL.join()
        except Exception:
            pass
        _GLOBAL_POOL = None

    if _GLOBAL_POOL is None:
        ctx = mp.get_context("spawn")
        _GLOBAL_POOL = ctx.Pool(
            processes=pool_size,
            initializer=worker_init,
        )
        _CURRENT_POOL_SIZE = pool_size

    return _GLOBAL_POOL


def close_worker_pool():
    """Explicitly closes and terminates the global worker pool."""
    global _GLOBAL_POOL, _CURRENT_POOL_SIZE
    if _GLOBAL_POOL is not None:
        try:
            _GLOBAL_POOL.close()
            _GLOBAL_POOL.terminate()
            _GLOBAL_POOL.join()
        except Exception:
            pass
        _GLOBAL_POOL = None
        _CURRENT_POOL_SIZE = None


def canonical_align_muzzle(
    crop_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: int = 448,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Align nostrils using local-background contrast; fall back to the mask principal axis."""
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

    smooth_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, min(h, w) / 18.0))
    dark_response = cv2.subtract(smooth_bg, gray)
    dark_response[valid == 0] = 0
    roi_h = max(1, int(h * 0.65))
    local = dark_response[:roi_h]
    local_mask = valid[:roi_h]
    roi_pixels = local[local_mask > 0]
    angle = 0.0
    method = "mask_principal_axis"
    points = []
    if roi_pixels.size and float(np.max(local)) >= 10.0:
        threshold = max(15.0, float(np.mean(roi_pixels) + 1.8 * np.std(roi_pixels)))
        threshold = min(threshold, max(15.0, 0.40 * float(np.max(local))))
        binary = np.where((local >= threshold) & (local_mask > 0), 255, 0).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask_area = max(1, int(np.count_nonzero(local_mask)))
        candidates = []
        for lab in range(1, count):
            area = int(stats[lab, cv2.CC_STAT_AREA])
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


def process_frame_stages_3_5(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Steps 3, 4, and 5 for a single candidate cattle muzzle frame.
    Runs inside a spawn process pool worker.
    """
    f_idx = task["frame_idx"]
    native_bgr = task["native_crop"]
    crop_mask = task.get("crop_mask")
    canonical_size = task.get("canonical_size", 448)
    method = task.get("method", "frangi")
    threshold_method = task.get("threshold_method", "adaptive")
    adaptive_block_size = task.get("adaptive_block_size", 15)
    adaptive_c = task.get("adaptive_c", 2)
    scales = task.get("scales")
    if scales is None:
        scales = [1.0, 1.5, 2.0]
    exclude_nostrils = task.get("exclude_nostrils", True)
    min_spur_length = task.get("min_spur_length", 8)
    min_component_size = task.get("min_component_size", 30)
    max_minutiae_points = task.get("max_minutiae_points", 250)

    # Step 3: Canonical Alignment & CLAHE local contrast normalization
    t_step3_start = time.perf_counter()
    aligned_bgr, aligned_mask, align_meta = canonical_align_muzzle(
        native_bgr, mask=crop_mask, target_size=canonical_size
    )
    angle_deg = float(align_meta.get("rotation_deg", 0.0))
    alignment_method = align_meta.get("alignment_method", "mask_principal_axis")
    upscaled_bgr = aligned_bgr
    mask_up = aligned_mask

    gray_up = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_up, (0, 0), sigmaX=1.5, sigmaY=1.5)
    gray_unsharp = cv2.addWeighted(gray_up, 1.5, blurred, -0.5, 0)
    clahe_filter = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe_filter.apply(gray_unsharp)
    t_step3_ms = (time.perf_counter() - t_step3_start) * 1000.0

    # Step 4: Multi-scale Black-Hat / Frangi groove filtering & adaptive threshold
    t_step4_start = time.perf_counter()
    enh_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    groove_res = extract_muzzle_grooves_pipeline(
        image_bgr=enh_bgr,
        mask=mask_up,
        method=method,
        threshold_method=threshold_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        scales=scales,
        exclude_nostrils=exclude_nostrils,
    )

    groove_response = groove_res["enhanced_response"]
    groove_mask = groove_res["groove_mask"]
    raw_skeleton = groove_res["groove_skeleton"]
    t_step4_ms = (time.perf_counter() - t_step4_start) * 1000.0

    # Step 5: Thin skeletonization, graph topology & minutiae extraction
    t_step5_start = time.perf_counter()
    pruned_skeleton = prune_skeleton(
        raw_skeleton,
        min_spur_length=min_spur_length,
        min_component_size=min_component_size,
    )

    graph_res = extract_skeleton_graph(
        pruned_skeleton,
        mask=groove_res.get("mask"),
        prune=False,  # Already pruned
    )
    desc = graph_res["descriptors"]
    overlay = visualize_groove_graph(upscaled_bgr, graph_res)

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

    ranked_minutiae = extract_minutiae_points(
        pruned_skeleton,
        max_points=max_minutiae_points,
    )
    minutiae = [
        {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "type": point["type"],
            "orientation": float(point["angle"]),
            "angle": float(point["angle"]),
            "weight": float(point["weight"]),
        }
        for point in ranked_minutiae
    ]

    matcher_mask = np.where(np.asarray(mask_up) > 0, 255, 0).astype(np.uint8) if mask_up is not None else np.full((canonical_size, canonical_size), 255, dtype=np.uint8)
    matcher_skeleton = np.where((pruned_skeleton > 0) & (matcher_mask > 0), 255, 0).astype(np.uint8)
    matcher_gray = enhanced_gray.copy()
    t_step5_ms = (time.perf_counter() - t_step5_start) * 1000.0

    return {
        "frame_idx": f_idx,
        "confidence": task.get("confidence", 0.0),
        "laplacian_var": task.get("laplacian_var", 0.0),
        "timestamp_sec": task.get("timestamp_sec", 0.0),
        "bbox": task.get("bbox", []),
        "native_bgr": native_bgr,
        "aligned_bgr": aligned_bgr,
        "aligned_mask": aligned_mask,
        "angle_deg": angle_deg,
        "alignment_method": alignment_method,
        "enhanced_gray": enhanced_gray,
        "groove_response": groove_response,
        "groove_mask": groove_mask,
        "pruned_skeleton": pruned_skeleton,
        "overlay": overlay,
        "params": params,
        "minutiae": minutiae,
        "matcher_gray": matcher_gray,
        "matcher_mask": matcher_mask,
        "matcher_skeleton": matcher_skeleton,
        "t_step3_ms": t_step3_ms,
        "t_step4_ms": t_step4_ms,
        "t_step5_ms": t_step5_ms,
    }


if __name__ == "__main__":
    pass
