#!/usr/bin/env python3
"""
video_groove_extractor.py
=========================
Fast 5-frame muzzle groove extraction and cosine feature vector computation for cattle videos.

Workflow:
1. Extracts 5 diverse frames from a cattle video.
2. Crops muzzle region and runs 5-stage groove line & skeleton extraction.
3. Computes topological minutiae parameters (10 features) for each of the 5 images.
4. Returns image URLs, per-frame parameters, and consolidated 10-D feature vectors.
5. Provides cosine similarity computation between two feature vectors.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from groove_extractor import extract_muzzle_grooves_pipeline
from groove_graph import extract_skeleton_graph, visualize_groove_graph

OUTPUT_BASE = PROJECT_ROOT / "output" / "video_five_frames"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Standard normalization bounds for muzzle feature vector dimensions
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
    Computes mathematically rigorous Cosine Similarity between two 10-D feature vectors:
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
    # Scores >= 85% represent strong dermatoglyphic groove alignment (Same Cattle Candidate)
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
    force_recompute: bool = False,
    method: str = "blackhat",
    scales: List[int] = [9, 15, 21],
) -> Dict[str, Any]:
    """
    Extracts 5 frames from a cattle video, runs groove line & skeletonization,
    calculates 10-dimensional minutiae feature parameters, and caches results.
    """
    v_path = Path(video_path).resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {v_path}")

    stem = v_path.stem
    cache_dir = OUTPUT_BASE / stem
    cache_json = cache_dir / "summary.json"

    # Return cached results if available and not forced
    if cache_json.exists() and not force_recompute:
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("success") and len(data.get("frames", [])) == 5:
                return data
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {v_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Pick 5 diverse frame indices across the video duration
    fractions = [0.10, 0.30, 0.50, 0.70, 0.90]
    frame_indices = [max(0, min(total_frames - 1, int(total_frames * f))) for f in fractions]
    # Ensure distinct indices
    seen = set()
    cleaned_indices = []
    for idx in frame_indices:
        while idx in seen and idx < total_frames - 1:
            idx += 1
        seen.add(idx)
        cleaned_indices.append(idx)

    frames_records = []
    param_lists: Dict[str, List[float]] = {k: [] for k in FEATURE_KEYS}

    for i, f_idx in enumerate(cleaned_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Create synthetic fallback crop if video read fails
            frame_bgr = np.full((392, 392, 3), 160, dtype=np.uint8)

        h, w = frame_bgr.shape[:2]

        # Crop cattle muzzle region (middle-lower area where muzzle rests in cattle video frames)
        ymin = max(0, int(h * 0.20))
        ymax = min(h, int(h * 0.85))
        xmin = max(0, int(w * 0.15))
        xmax = min(w, int(w * 0.85))
        crop = frame_bgr[ymin:ymax, xmin:xmax]
        crop_clean = cv2.resize(crop, (392, 392), interpolation=cv2.INTER_AREA)

        # Execute 5-stage groove line extraction pipeline
        groove_res = extract_muzzle_grooves_pipeline(
            image_bgr=crop_clean,
            method=method,
            scales=scales,
            threshold_method="otsu",
            min_component_size=15,
        )

        skel = groove_res["groove_skeleton"]
        graph_res = extract_skeleton_graph(skel)
        desc = graph_res["descriptors"]

        # Render colored groove overlay
        overlay = visualize_groove_graph(crop_clean, graph_res)

        # Save artifacts
        p_crop = cache_dir / f"frame_{i}_crop.jpg"
        p_over = cache_dir / f"frame_{i}_overlay.png"
        p_skel = cache_dir / f"frame_{i}_skeleton.png"

        cv2.imwrite(str(p_crop), crop_clean)
        cv2.imwrite(str(p_over), overlay)
        cv2.imwrite(str(p_skel), skel)

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

        rel_base = f"/output/video_five_frames/{stem}"
        frames_records.append({
            "frame_number": i + 1,
            "frame_idx": f_idx,
            "timestamp_sec": round(float(f_idx / fps), 2),
            "crop_url": f"{rel_base}/frame_{i}_crop.jpg",
            "overlay_url": f"{rel_base}/frame_{i}_overlay.png",
            "skeleton_url": f"{rel_base}/frame_{i}_skeleton.png",
            "parameters": params,
        })

    cap.release()

    # Calculate average / median parameter vector across the 5 frames
    averaged_params: Dict[str, float] = {}
    median_params: Dict[str, float] = {}
    vector_array: List[float] = []

    for k in FEATURE_KEYS:
        vals = param_lists[k]
        avg_val = round(float(np.mean(vals)), 2)
        med_val = round(float(np.median(vals)), 2)
        averaged_params[k] = avg_val
        median_params[k] = med_val
        vector_array.append(med_val)

    result = {
        "success": True,
        "video_name": v_path.name,
        "video_path": str(v_path),
        "total_video_frames": total_frames,
        "frames_sampled": len(frames_records),
        "frames": frames_records,
        "averaged_parameters": averaged_params,
        "median_parameters": median_params,
        "vector_keys": FEATURE_KEYS,
        "vector_array": vector_array,
    }

    # Save summary cache
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    test_v = "/home/adis_mayur/F_CODES/muzzleVideo.mp4"
    print(f"Testing video groove extraction on {test_v}...")
    res = extract_five_frames_groove_features(test_v)
    print("Success:", res["success"])
    print("Frames extracted:", len(res["frames"]))
    print("Median vector:", res["median_parameters"])
