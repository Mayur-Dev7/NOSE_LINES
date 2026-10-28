#!/usr/bin/env python3
"""
Cattle Muzzle Biometric Matcher & Verification Pipeline
======================================================
Based on /home/adis_mayur/Downloads/statusprint.py:
- Extracts high-quality muzzle crops from 2 different cattle videos / images
- Upscales with edge-preserving super-resolution (Lanczos-4 Sinc / Bicubic)
- Converts to high-contrast Grayscale with CLAHE dermatoglyphic ridge enhancement
- Performs ORB keypoint feature extraction and Lowe-ratio Hamming matching
- Evaluates against DUPLICATE_THRESHOLD (35 matches) to decide:
    SAME CATTLE (Duplicate / Match) vs DIFFERENT CATTLE (Distinct / Verified)
- Generates side-by-side keypoint visual match diagrams
"""

import os
import sys
import json
import time
import uuid
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import cv2
import numpy as np

# Ensure mrcnn is in path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from groove_extractor import extract_muzzle_grooves_pipeline
from groove_graph import extract_skeleton_graph

logger = logging.getLogger("muzzle_matcher")

# Biometric matching thresholds & canonical dimensions
DEFAULT_DUPLICATE_THRESHOLD = 35
DEFAULT_MATCH_THRESHOLD = 0.45
CANONICAL_SIZE = (448, 448)
CONFIG_PATH = SCRIPT_DIR / "matcher_config.json"
MATCH_VISUAL_DIR = SCRIPT_DIR / "output" / "match_visualizations"
MATCH_VISUAL_DIR.mkdir(parents=True, exist_ok=True)


def to_output_url(file_path: Optional[str]) -> Optional[str]:
    """Converts a local file path inside the output directory to an API-relative URL."""
    if not file_path:
        return None
    try:
        p = Path(file_path).resolve()
        parts = p.parts
        if "output" in parts:
            idx = parts.index("output")
            rel_parts = parts[idx + 1:]
            return "/output/" + "/".join(rel_parts)
    except Exception:
        pass
    return f"/output/{Path(file_path).name}"

# Early CPU affinity & thread allocation before native libraries load
if "--cpu-cores" in sys.argv:
    try:
        _idx = sys.argv.index("--cpu-cores")
        if _idx + 1 < len(sys.argv):
            _early_cores = int(sys.argv[_idx + 1])
            if _early_cores > 0:
                os.environ["OMP_NUM_THREADS"] = str(_early_cores)
                os.environ["TF_NUM_INTRAOP_THREADS"] = str(_early_cores)
                os.environ["TF_NUM_INTEROP_THREADS"] = str(min(_early_cores, 2))
                os.environ["MKL_NUM_THREADS"] = str(_early_cores)
                os.environ["OPENBLAS_NUM_THREADS"] = str(_early_cores)
                os.environ["NUMEXPR_NUM_THREADS"] = str(_early_cores)
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(_early_cores)
                if hasattr(os, "sched_setaffinity"):
                    _max_c = os.cpu_count() or _early_cores
                    os.sched_setaffinity(0, set(range(min(_early_cores, _max_c))))
    except Exception:
        pass

# Statusprint threshold (from /home/adis_mayur/Downloads/statusprint.py)
DEFAULT_DUPLICATE_THRESHOLD = 35


def compute_image_quality(image_bgr: np.ndarray, confidence: float = 1.0) -> Dict[str, float]:
    """
    Computes quality metrics for a cattle muzzle crop:
    - Sharpness via Laplacian variance
    - Resolution / area
    - Contrast via standard deviation
    - Composite quality score (0 - 100)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_dev = float(np.std(gray))
    h, w = gray.shape[:2]
    area = h * w

    # Normalized scores
    sharpness_score = min(100.0, (laplacian_var / 500.0) * 100.0)
    contrast_score = min(100.0, (std_dev / 64.0) * 100.0)
    res_score = min(100.0, (min(h, w) / 224.0) * 100.0)
    conf_score = confidence * 100.0

    # Composite score
    composite = (sharpness_score * 0.35) + (contrast_score * 0.25) + (conf_score * 0.25) + (res_score * 0.15)

    return {
        "composite": round(composite, 1),
        "sharpness": round(laplacian_var, 1),
        "contrast": round(std_dev, 1),
        "resolution": f"{w}x{h}",
        "area": area,
        "confidence": round(confidence * 100, 1),
    }


def enhance_muzzle_image(
    image_bgr: np.ndarray,
    method: str = "lanczos4",
    scale_factor: int = 2,
    unsharp: bool = True,
    clahe: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upscales cattle muzzle image preserving fine bead/ridge dermatoglyphics,
    then transforms to contrast-enhanced grayscale.

    Args:
        image_bgr: Input BGR image
        method: 'lanczos4' (recommended, sinc 8-point), 'bicubic', or 'bilinear'
        scale_factor: 2 (e.g. 224x224 -> 448x448) or 3
        unsharp: Apply unsharp masking filter to accentuate ridge edges
        clahe: Apply Contrast Limited Adaptive Histogram Equalization

    Returns:
        (enhanced_gray, upscaled_bgr)
    """
    interp_map = {
        "lanczos4": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
    }
    interp = interp_map.get(method.lower(), cv2.INTER_LANCZOS4)

    h, w = image_bgr.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # 1. Super-resolution / Edge-preserving Upscale
    upscaled_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Unsharp Masking for Dermatoglyphic Groove Sharpening
    if unsharp:
        # Subtle Gaussian blur subtraction to boost micro-edges
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
        # unsharp mask = original + amount * (original - blurred)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 4. CLAHE (Local Contrast Normalization)
    if clahe:
        # clipLimit=2.5 prevents noise amplification while bringing out ridge beads
        clahe_filter = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe_filter.apply(gray)

    return gray, upscaled_bgr


def compare_two_muzzles(
    gray1: np.ndarray,
    gray2: np.ndarray,
    nfeatures: int = 1000,
    threshold: int = DEFAULT_DUPLICATE_THRESHOLD,
    bgr1: Optional[np.ndarray] = None,
    bgr2: Optional[np.ndarray] = None,
    save_visual_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compares two grayscale muzzle prints using statusprint.py's ORB + BFMatcher logic.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return {
            "good_matches": 0,
            "threshold": threshold,
            "is_match": False,
            "verdict": "NO_MATCH",
            "similarity_percent": 0.0,
            "kp1_count": len(kp1) if kp1 else 0,
            "kp2_count": len(kp2) if kp2 else 0,
            "visual_path": None,
        }

    # KNN matching with k=2 & Lowe's ratio test (0.7) exactly like statusprint.py
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        matches = []

    good_pairs = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
            good_pairs.append(pair[0])

    good_matches = len(good_pairs)
    is_match = good_matches >= threshold
    verdict = "MATCH_SAME_CATTLE" if is_match else "DIFFERENT_CATTLE"

    # Similarity percentage heuristic (35 matches is the decision boundary at ~70%)
    if is_match:
        # 35 matches -> 75%, 65+ matches -> 98%
        similarity_percent = min(99.5, 70.0 + ((good_matches - threshold) / 30.0) * 28.0)
    else:
        # Below threshold
        similarity_percent = min(68.0, (good_matches / float(threshold)) * 68.0)

    # Generate Visual Match Diagram if requested
    visual_saved = None
    if save_visual_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_visual_path)), exist_ok=True)
        img_left = bgr1 if bgr1 is not None else cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        img_right = bgr2 if bgr2 is not None else cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

        # Draw top 40 best matching lines
        top_matches = sorted(good_pairs, key=lambda m: m.distance)[:40]
        match_canvas = cv2.drawMatches(
            img_left, kp1,
            img_right, kp2,
            top_matches,
            None,
            matchColor=(0, 255, 0) if is_match else (0, 165, 255),
            singlePointColor=(100, 100, 100),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Add verdict banner on top of the canvas
        banner_h = 45
        banner = np.zeros((banner_h, match_canvas.shape[1], 3), dtype=np.uint8)
        color = (34, 197, 94) if is_match else (239, 68, 68)
        label = f"VERDICT: {verdict.replace('_', ' ')} | Good Matches: {good_matches} (Threshold: {threshold}) | Sim: {similarity_percent:.1f}%"
        cv2.putText(banner, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        final_composite = np.vstack([banner, match_canvas])
        cv2.imwrite(save_visual_path, final_composite)
        visual_saved = save_visual_path

    return {
        "good_matches": good_matches,
        "threshold": threshold,
        "is_match": is_match,
        "verdict": verdict,
        "similarity_percent": round(similarity_percent, 1),
        "kp1_count": len(kp1),
        "kp2_count": len(kp2),
        "visual_path": visual_saved,
    }


def load_matcher_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Reads matcher_config.json if it exists.
    If not found or uncalibrated, returns {"calibrated": False, "status": "UNCALIBRATED", "threshold": None}.
    """
    p = Path(config_path) if config_path else CONFIG_PATH
    if p.exists() and p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("calibrated", False):
                if "status" not in data:
                    data["status"] = "CALIBRATED"
                return data
        except Exception as e:
            logger.warning(f"Failed reading matcher config {p}: {e}")
    return {"calibrated": False, "status": "UNCALIBRATED", "threshold": None}


def match_sift_ransac(
    gray1: np.ndarray,
    gray2: np.ndarray,
    nfeatures: int = 1500,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    kp1: Optional[List[Any]] = None,
    des1: Optional[np.ndarray] = None,
    kp2: Optional[List[Any]] = None,
    des2: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Method A: SIFT Keypoint Matching with ORB Fallback + RANSAC Affine Alignment.
    
    Inputs:
        gray1, gray2: Canonical 448x448 CLAHE grayscale muzzle crops.
        nfeatures: Maximum number of keypoints to detect (default 1500).
        ratio_thresh: Lowe's ratio test threshold (0.75).
        ransac_thresh: RANSAC reprojection error threshold in pixels (5.0).
        kp1, des1, kp2, des2: Optional precomputed keypoints and descriptors for reuse.
        
    Returns:
        Dict with inlier_count, inlier_ratio, keypoint_overlap_ratio, score_method_a, etc.
    """
    if gray1.ndim == 3:
        gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
    if gray2.ndim == 3:
        gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

    if gray1.shape[:2] != CANONICAL_SIZE:
        gray1 = cv2.resize(gray1, CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4)
    if gray2.shape[:2] != CANONICAL_SIZE:
        gray2 = cv2.resize(gray2, CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4)

    detector_type = "sift"

    # Step 1: Keypoint extraction with SIFT
    if kp1 is None or des1 is None:
        try:
            sift = cv2.SIFT_create(nfeatures=nfeatures)
            kp1, des1 = sift.detectAndCompute(gray1, None)
        except Exception:
            kp1, des1 = [], None

    if kp2 is None or des2 is None:
        try:
            sift = cv2.SIFT_create(nfeatures=nfeatures)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        except Exception:
            kp2, des2 = [], None

    # Step 2: Fallback to ORB if SIFT descriptors are empty/none/too sparse
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        try:
            orb = cv2.ORB_create(nfeatures=nfeatures)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            detector_type = "orb"
        except Exception:
            pass

    # Safety check if still empty
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return {
            "detector": detector_type,
            "kp1_count": len(kp1) if kp1 else 0,
            "kp2_count": len(kp2) if kp2 else 0,
            "good_matches_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "keypoint_overlap_ratio": 0.0,
            "score_method_a": 0.0,
            "transform_matrix": None,
            "inliers_mask": [],
            "good_matches": [],
            "kp1": kp1 or [],
            "kp2": kp2 or [],
        }

    # Step 3: KNN matching with k=2 & Lowe's ratio test
    if detector_type == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        raw_matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        raw_matches = []

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good_matches.append(pair[0])

    # Step 4: RANSAC partial affine fit (rotation + translation + uniform scale)
    inlier_count = 0
    inliers_mask = []
    transform_matrix = None

    if len(good_matches) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
        if M is None or mask is None:
            M, mask = cv2.estimateAffine2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh,
            )
        if mask is not None:
            inliers_mask = mask.ravel().astype(int).tolist()
            inlier_count = int(sum(inliers_mask))
            transform_matrix = M
        else:
            inliers_mask = [0] * len(good_matches)
            inlier_count = 0
    elif len(good_matches) > 0:
        inliers_mask = [1] * len(good_matches)
        inlier_count = len(good_matches)

    # Step 5: Metric calculations
    good_count = len(good_matches)
    inlier_ratio = float(inlier_count / max(1, good_count))
    keypoint_overlap_ratio = float(inlier_count / max(1, min(len(kp1), len(kp2))))
    score_method_a = float(min(1.0, (inlier_count / 50.0) * 0.7 + inlier_ratio * 0.3))

    return {
        "detector": detector_type,
        "kp1_count": len(kp1),
        "kp2_count": len(kp2),
        "good_matches_count": good_count,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "keypoint_overlap_ratio": round(keypoint_overlap_ratio, 4),
        "score_method_a": round(score_method_a, 4),
        "transform_matrix": transform_matrix.tolist() if transform_matrix is not None else None,
        "inliers_mask": inliers_mask,
        "good_matches": good_matches,
        "kp1": kp1,
        "kp2": kp2,
    }


def extract_minutiae_points(
    skeleton: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_minutiae: int = 250,
) -> List[Dict[str, Any]]:
    """
    Extracts topological minutiae points (endpoints & junctions) with spatial coordinates,
    type, and orientation angle in degrees [0, 360).
    """
    if skeleton is None or skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return []

    graph_res = extract_skeleton_graph(skeleton, mask=mask)
    nodes = graph_res.get("nodes", {})
    branches = graph_res.get("branches", [])

    # Map node IDs to branches for orientation estimation
    node_to_branches: Dict[int, List[Dict[str, Any]]] = {}
    for b in branches:
        u = b.get("start_node")
        v = b.get("end_node")
        if u is not None:
            node_to_branches.setdefault(u, []).append(b)
        if v is not None and v != u:
            node_to_branches.setdefault(v, []).append(b)

    H, W = skeleton.shape[:2]
    minutiae = []

    for nid, node in nodes.items():
        pos = node.get("pos", (0.0, 0.0))
        x, y = float(pos[0]), float(pos[1])
        ntype = str(node.get("type", "endpoint"))

        # Determine orientation angle: check node attributes or connected branches
        b_list = node_to_branches.get(nid, [])
        if "orientation" in node and node["orientation"] is not None:
            orient_deg = float(node["orientation"])
        elif b_list:
            best_b = max(b_list, key=lambda b: b.get("arc_length", 0.0))
            coords = best_b.get("pixel_coords", [])
            if best_b.get("start_node") == nid and len(coords) >= 2:
                step = min(4, len(coords) - 1)
                py, px = coords[step]
                orient_deg = float(math.degrees(math.atan2(py - y, px - x)))
            elif best_b.get("end_node") == nid and len(coords) >= 2:
                step = max(0, len(coords) - 5)
                py, px = coords[step]
                orient_deg = float(math.degrees(math.atan2(py - y, px - x)))
            else:
                orient_deg = float(best_b.get("orientation_deg", 0.0))
        else:
            orient_deg = 0.0

        orient_deg = float(orient_deg % 360.0)
        minutiae.append({
            "x": round(x, 2),
            "y": round(y, 2),
            "type": ntype,
            "orientation": round(orient_deg, 2),
        })

    # Limit to top minutiae by centrality if dense
    if len(minutiae) > max_minutiae:
        cx, cy = W / 2.0, H / 2.0
        minutiae.sort(key=lambda m: (m["x"] - cx) ** 2 + (m["y"] - cy) ** 2)
        minutiae = minutiae[:max_minutiae]

    return minutiae


def match_minutiae(
    minutiae_a: Union[List[Dict[str, Any]], np.ndarray],
    minutiae_b: Union[List[Dict[str, Any]], np.ndarray],
    spatial_threshold: float = 15.0,
    angular_threshold: float = 35.0,
    max_trials: int = 120,
) -> Dict[str, Any]:
    """
    Method B: Minutiae Matching from Skeleton Endpoints / Junctions using RANSAC.
    
    Aligns minutiae sets A and B with rigid/partial affine transformation,
    and counts pairs satisfying distance < spatial_threshold and angle_diff < angular_threshold.
    """
    if isinstance(minutiae_a, np.ndarray):
        minutiae_a = extract_minutiae_points(minutiae_a)
    if isinstance(minutiae_b, np.ndarray):
        minutiae_b = extract_minutiae_points(minutiae_b)

    na = len(minutiae_a) if minutiae_a else 0
    nb = len(minutiae_b) if minutiae_b else 0

    if na < 2 or nb < 2:
        return {
            "matched_count": 0,
            "matched_fraction": 0.0,
            "score": 0.0,
            "score_method_b": 0.0,
            "minutiae_a_count": na,
            "minutiae_b_count": nb,
            "transform_matrix": None,
        }

    pts_a = np.float32([[m["x"], m["y"]] for m in minutiae_a]).reshape(-1, 1, 2)
    pts_b = np.float32([[m["x"], m["y"]] for m in minutiae_b]).reshape(-1, 1, 2)
    types_a = [m.get("type", "endpoint") for m in minutiae_a]
    types_b = [m.get("type", "endpoint") for m in minutiae_b]

    # Handle radians or degrees
    raw_deg_a = [float(m.get("orientation", 0.0)) for m in minutiae_a]
    raw_deg_b = [float(m.get("orientation", 0.0)) for m in minutiae_b]
    if max(raw_deg_a or [0]) <= 2.0 * math.pi + 0.1 and max(raw_deg_a or [0]) > 0:
        raw_deg_a = [math.degrees(a) for a in raw_deg_a]
    if max(raw_deg_b or [0]) <= 2.0 * math.pi + 0.1 and max(raw_deg_b or [0]) > 0:
        raw_deg_b = [math.degrees(b) for b in raw_deg_b]

    deg_a = np.array(raw_deg_a, dtype=np.float32)
    deg_b = np.array(raw_deg_b, dtype=np.float32)

    def evaluate_transform(M: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
        trans_a = cv2.transform(pts_a, M).reshape(-1, 2)
        rot_deg = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        trans_deg = (deg_a + rot_deg) % 360.0

        # Vectorized Euclidean distance matrix (Na, Nb)
        dx = pts_b[:, 0, 0][None, :] - trans_a[:, 0][:, None]
        dy = pts_b[:, 0, 1][None, :] - trans_a[:, 1][:, None]
        dist_sq = dx * dx + dy * dy

        # Angular difference matrix (Na, Nb)
        adiff = np.abs(trans_deg[:, None] - deg_b[None, :]) % 360.0
        adiff = np.minimum(adiff, 360.0 - adiff)

        type_match = np.array(types_a)[:, None] == np.array(types_b)[None, :]
        valid = (dist_sq < (spatial_threshold ** 2)) & (adiff < angular_threshold) & type_match

        valid_indices = np.argwhere(valid)
        if len(valid_indices) == 0:
            return 0, []

        dists = dist_sq[valid_indices[:, 0], valid_indices[:, 1]]
        order = np.argsort(dists)

        used_a: Set[int] = set()
        used_b: Set[int] = set()
        matched: List[Tuple[int, int]] = []
        for idx in order:
            ia, ib = int(valid_indices[idx, 0]), int(valid_indices[idx, 1])
            if ia not in used_a and ib not in used_b:
                used_a.add(ia)
                used_b.add(ib)
                matched.append((ia, ib))

        return len(matched), matched

    # 1. Baseline: evaluate identity transform
    best_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    best_count, best_matched = evaluate_transform(best_M)

    # 2. RANSAC candidate pair alignment
    eps_a = [i for i, t in enumerate(types_a) if t == "endpoint"]
    eps_b = [i for i, t in enumerate(types_b) if t == "endpoint"]
    juncs_a = [i for i, t in enumerate(types_a) if t == "junction"]
    juncs_b = [i for i, t in enumerate(types_b) if t == "junction"]

    rng = np.random.RandomState(42)
    trials = 0

    while trials < max_trials:
        trials += 1
        # Prefer matching same-type candidate pairs
        use_eps = (len(eps_a) >= 2 and len(eps_b) >= 2) and (rng.rand() > 0.5 or len(juncs_a) < 2)
        if use_eps:
            pool_a, pool_b = eps_a, eps_b
        elif len(juncs_a) >= 2 and len(juncs_b) >= 2:
            pool_a, pool_b = juncs_a, juncs_b
        else:
            pool_a, pool_b = list(range(na)), list(range(nb))

        ia1, ia2 = rng.choice(pool_a, 2, replace=False)
        ib1, ib2 = rng.choice(pool_b, 2, replace=False)

        d_a = math.hypot(pts_a[ia1, 0, 0] - pts_a[ia2, 0, 0], pts_a[ia1, 0, 1] - pts_a[ia2, 0, 1])
        d_b = math.hypot(pts_b[ib1, 0, 0] - pts_b[ib2, 0, 0], pts_b[ib1, 0, 1] - pts_b[ib2, 0, 1])
        if d_a < 15.0 or d_b < 15.0:
            continue
        ratio = d_a / max(1.0, d_b)
        if ratio < 0.70 or ratio > 1.40:
            continue

        pair_a = np.float32([[pts_a[ia1, 0, 0], pts_a[ia1, 0, 1]], [pts_a[ia2, 0, 0], pts_a[ia2, 0, 1]]]).reshape(-1, 1, 2)
        pair_b = np.float32([[pts_b[ib1, 0, 0], pts_b[ib1, 0, 1]], [pts_b[ib2, 0, 0], pts_b[ib2, 0, 1]]]).reshape(-1, 1, 2)

        M, _ = cv2.estimateAffinePartial2D(pair_a, pair_b)
        if M is None:
            continue

        scale = math.hypot(M[0, 0], M[0, 1])
        if scale < 0.65 or scale > 1.50:
            continue

        cnt, matched = evaluate_transform(M)
        if cnt > best_count:
            best_count = cnt
            best_matched = matched
            best_M = M

    # 3. Refinement: re-estimate partial affine from all matched inlier pairs
    if best_count >= 4 and best_matched:
        match_a_idx = [p[0] for p in best_matched]
        match_b_idx = [p[1] for p in best_matched]
        ref_pts_a = pts_a[match_a_idx]
        ref_pts_b = pts_b[match_b_idx]
        M_ref, _ = cv2.estimateAffinePartial2D(ref_pts_a, ref_pts_b)
        if M_ref is not None:
            cnt_ref, matched_ref = evaluate_transform(M_ref)
            if cnt_ref >= best_count:
                best_count = cnt_ref
                best_matched = matched_ref
                best_M = M_ref

    matched_fraction = float(best_count / max(1, min(na, nb)))
    score_b = round(matched_fraction, 4)

    return {
        "matched_count": best_count,
        "matched_fraction": score_b,
        "score": score_b,
        "score_method_b": score_b,
        "minutiae_a_count": na,
        "minutiae_b_count": nb,
        "transform_matrix": best_M.tolist() if best_M is not None else None,
    }


def _resolve_frame_data(
    frame_obj: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    Resolves arbitrary frame representation into:
    (canonical_gray_448, optional_skeleton_img, minutiae_list).
    """
    gray: Optional[np.ndarray] = None
    skel: Optional[np.ndarray] = None
    minutiae: List[Dict[str, Any]] = []

    if isinstance(frame_obj, np.ndarray):
        gray = frame_obj
    elif isinstance(frame_obj, (str, Path)):
        p = Path(frame_obj)
        if p.exists():
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    elif isinstance(frame_obj, dict):
        # 1. Grayscale image retrieval
        for k in ["enhanced_gray", "enhanced_image", "cropped_bgr", "crop"]:
            if isinstance(frame_obj.get(k), np.ndarray):
                gray = frame_obj[k]
                break

        if gray is None:
            for pk in ["enhanced_path", "crop_path"]:
                pth = frame_obj.get(pk)
                if pth and Path(pth).exists():
                    gray = cv2.imread(str(pth), cv2.IMREAD_GRAYSCALE)
                    break

        if gray is None:
            for uk in ["enhanced_url", "crop_url"]:
                u = frame_obj.get(uk)
                if u and isinstance(u, str):
                    rel = u.lstrip("/")
                    cand = SCRIPT_DIR / rel
                    if cand.exists():
                        gray = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                        break

        # 2. Skeleton retrieval
        for sk in ["groove_skeleton", "skeleton"]:
            if isinstance(frame_obj.get(sk), np.ndarray):
                skel = frame_obj[sk]
                break

        if skel is None:
            sp = frame_obj.get("skeleton_path")
            if sp and Path(sp).exists():
                skel = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)

        if skel is None:
            su = frame_obj.get("skeleton_url")
            if su and isinstance(su, str):
                rel = su.lstrip("/")
                cand = SCRIPT_DIR / rel
                if cand.exists():
                    skel = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)

        # 3. Minutiae retrieval
        if isinstance(frame_obj.get("minutiae"), list):
            minutiae = frame_obj["minutiae"]

    # Fallback placeholder if no image found
    if gray is None:
        gray = np.zeros(CANONICAL_SIZE, dtype=np.uint8)

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if gray.shape[:2] != CANONICAL_SIZE:
        gray = cv2.resize(gray, CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Extract minutiae if not pre-provided
    if not minutiae:
        if skel is not None:
            minutiae = extract_minutiae_points(skel)
        else:
            # Extract groove skeleton from grayscale
            try:
                bgr_equiv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                groove_out = extract_muzzle_grooves_pipeline(
                    bgr_equiv,
                    method="frangi",
                    threshold_method="adaptive",
                    exclude_nostrils=True,
                )
                skel = groove_out.get("groove_skeleton")
                if skel is not None:
                    minutiae = extract_minutiae_points(skel)
            except Exception as e:
                logger.warning(f"Failed fallback skeleton extraction: {e}")

    return gray, skel, minutiae


def match_muzzle_frame_sets(
    frames_a: List[Any],
    frames_b: List[Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    5x5 All-Pairs Cattle Muzzle Biometric Comparison:
    - Compares all pairs (i, j) between Video A and Video B frame sets.
    - Pair composite score = 0.65 * score_method_a + 0.35 * score_method_b.
    - Computes Best Pair score and Median score across all 25 pairs.
    - Overall Composite Score = 0.7 * best_score + 0.3 * median_score.
    - Generates side-by-side keypoint visualization image for the best matching pair.
    - Evaluates against calibrated threshold or default.
    """
    if not frames_a or not frames_b:
        raise ValueError("Both frames_a and frames_b must contain at least 1 frame.")

    out_dir = Path(output_dir) if output_dir else (SCRIPT_DIR / "output")
    vis_dir = out_dir / "match_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Pre-extract features for all frames in A and B (10 extractions instead of 50)
    data_a = []
    for fa in frames_a:
        g, sk, mins = _resolve_frame_data(fa)
        sift = cv2.SIFT_create(nfeatures=1500)
        try:
            kp, des = sift.detectAndCompute(g, None)
        except Exception:
            kp, des = [], None
        data_a.append({"gray": g, "kp": kp, "des": des, "minutiae": mins})

    data_b = []
    for fb in frames_b:
        g, sk, mins = _resolve_frame_data(fb)
        sift = cv2.SIFT_create(nfeatures=1500)
        try:
            kp, des = sift.detectAndCompute(g, None)
        except Exception:
            kp, des = [], None
        data_b.append({"gray": g, "kp": kp, "des": des, "minutiae": mins})

    n_a = len(data_a)
    n_b = len(data_b)

    pair_scores: List[List[float]] = [[0.0 for _ in range(n_b)] for _ in range(n_a)]
    pair_details: Dict[Tuple[int, int], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    best_pair_dict = {
        "frame_a_idx": 0,
        "frame_b_idx": 0,
        "score": -1.0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "matched_minutiae_fraction": 0.0,
        "score_method_a": 0.0,
        "score_method_b": 0.0,
    }
    best_res_a = {}
    best_res_b = {}
    all_scores: List[float] = []

    # Evaluate all pairs (e.g. 5x5 = 25 pairs)
    for i in range(n_a):
        for j in range(n_b):
            res_a = match_sift_ransac(
                gray1=data_a[i]["gray"],
                gray2=data_b[j]["gray"],
                kp1=data_a[i]["kp"],
                des1=data_a[i]["des"],
                kp2=data_b[j]["kp"],
                des2=data_b[j]["des"],
            )

            res_b = match_minutiae(
                minutiae_a=data_a[i]["minutiae"],
                minutiae_b=data_b[j]["minutiae"],
            )

            pair_comp = round(0.65 * res_a["score_method_a"] + 0.35 * res_b["score_method_b"], 4)
            pair_scores[i][j] = pair_comp
            all_scores.append(pair_comp)
            pair_details[(i, j)] = (res_a, res_b)

            if pair_comp > best_pair_dict["score"]:
                best_pair_dict = {
                    "frame_a_idx": i,
                    "frame_b_idx": j,
                    "score": pair_comp,
                    "inlier_count": res_a["inlier_count"],
                    "inlier_ratio": res_a["inlier_ratio"],
                    "matched_minutiae_fraction": res_b["matched_fraction"],
                    "score_method_a": res_a["score_method_a"],
                    "score_method_b": res_b["score_method_b"],
                }
                best_res_a = res_a
                best_res_b = res_b

    best_score = float(max(0.0, best_pair_dict["score"]))
    median_score = float(np.median(all_scores)) if all_scores else 0.0
    composite_score = round(0.70 * best_score + 0.30 * median_score, 4)

    # Determine threshold and verdict
    cfg = load_matcher_config()
    if cfg.get("calibrated") and cfg.get("threshold") is not None:
        threshold = float(cfg["threshold"])
        threshold_calibrated = True
    else:
        threshold = DEFAULT_MATCH_THRESHOLD
        threshold_calibrated = False

    is_match = bool(composite_score >= threshold)
    verdict = "MATCH_SAME_CATTLE" if is_match else "DIFFERENT_CATTLE"

    # Generate side-by-side keypoint match visualization image for best pair
    bi = best_pair_dict["frame_a_idx"]
    bj = best_pair_dict["frame_b_idx"]
    gray_left = data_a[bi]["gray"]
    gray_right = data_b[bj]["gray"]

    bgr_left = cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR)
    bgr_right = cv2.cvtColor(gray_right, cv2.COLOR_GRAY2BGR)

    good_matches = best_res_a.get("good_matches", [])
    inliers_mask = best_res_a.get("inliers_mask", [])

    inlier_matches = [
        m for m, mask_v in zip(good_matches, inliers_mask) if mask_v == 1
    ]
    draw_matches = inlier_matches[:50] if inlier_matches else good_matches[:35]

    match_canvas = cv2.drawMatches(
        bgr_left, best_res_a.get("kp1", []),
        bgr_right, best_res_a.get("kp2", []),
        draw_matches,
        None,
        matchColor=(34, 197, 94) if is_match else (0, 165, 255),
        singlePointColor=(110, 110, 110),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Draw header banner on top
    banner_h = 56
    banner = np.zeros((banner_h, match_canvas.shape[1], 3), dtype=np.uint8)
    banner_color = (34, 197, 94) if is_match else (50, 50, 240)

    title_str = (
        f"VERDICT: {verdict.replace('_', ' ')}  |  "
        f"Composite Score: {composite_score:.3f} (Threshold: {threshold:.3f})"
    )
    metrics_str = (
        f"Best Pair: (Frame A{bi}, Frame B{bj})  |  "
        f"SIFT Inliers: {best_pair_dict['inlier_count']} ({best_pair_dict['inlier_ratio']*100:.1f}%)  |  "
        f"Minutiae Match: {best_pair_dict['matched_minutiae_fraction']*100:.1f}%"
    )

    cv2.putText(banner, title_str, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, banner_color, 2, cv2.LINE_AA)
    cv2.putText(banner, metrics_str, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    composite_vis = np.vstack([banner, match_canvas])
    vis_filename = f"match_pair_A{bi}_B{bj}_{uuid.uuid4().hex[:8]}.jpg"
    vis_path = vis_dir / vis_filename
    cv2.imwrite(str(vis_path), composite_vis)

    visual_url = to_output_url(str(vis_path)) or f"/output/match_visualizations/{vis_filename}"

    return {
        "best_pair": best_pair_dict,
        "median_score": round(median_score, 4),
        "best_score": round(best_score, 4),
        "composite_score": round(composite_score, 4),
        "inlier_count": best_pair_dict["inlier_count"],
        "inlier_ratio": best_pair_dict["inlier_ratio"],
        "matched_minutiae_fraction": best_pair_dict["matched_minutiae_fraction"],
        "method_a": {
            "inlier_count": best_pair_dict["inlier_count"],
            "inlier_ratio": best_pair_dict["inlier_ratio"],
            "score": best_pair_dict["score_method_a"],
        },
        "method_b": {
            "matched_count": best_res_b.get("matched_count", 0),
            "matched_fraction": best_pair_dict["matched_minutiae_fraction"],
            "score": best_pair_dict["score_method_b"],
        },
        "visual_url": visual_url,
        "pair_scores": pair_scores,
        "is_match": is_match,
        "verdict": verdict,
        "threshold": threshold,
        "threshold_calibrated": threshold_calibrated,
    }


def match_two_videos_end_to_end(
    video_a_path: Union[str, Path],
    video_b_path: Union[str, Path],
    filter_method: str = "frangi",
    threshold_method: str = "adaptive",
    adaptive_block_size: int = 15,
    adaptive_c: int = 2,
    exclude_nostrils: bool = True,
    device: str = "cpu",
    cpu_cores: int = 6,
    output_dir: Optional[Union[str, Path]] = None,
    log_callback: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    End-to-end cattle video biometric verification:
    1. Extracts 5 muzzle stage frames from Video A using extract_five_muzzle_stages.
    2. Extracts 5 muzzle stage frames from Video B using extract_five_muzzle_stages.
    3. Runs 5x5 All-Pairs Biometric Comparison (SIFT/RANSAC + Minutiae Matching).
    """
    from video_groove_extractor import extract_five_muzzle_stages

    out_dir = Path(output_dir) if output_dir else (SCRIPT_DIR / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_callback:
        log_callback(f"Extracting 5 muzzle stages from Video A: {Path(video_a_path).name}...")

    res_a = extract_five_muzzle_stages(
        video_path=video_a_path,
        method=filter_method,
        threshold_method=threshold_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        exclude_nostrils=exclude_nostrils,
        device=device,
        cpu_cores=cpu_cores,
        log_callback=log_callback,
    )

    if log_callback:
        log_callback(f"Extracting 5 muzzle stages from Video B: {Path(video_b_path).name}...")

    res_b = extract_five_muzzle_stages(
        video_path=video_b_path,
        method=filter_method,
        threshold_method=threshold_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        exclude_nostrils=exclude_nostrils,
        device=device,
        cpu_cores=cpu_cores,
        log_callback=log_callback,
    )

    frames_a = res_a.get("frames", [])
    frames_b = res_b.get("frames", [])

    if not frames_a:
        raise ValueError(f"No muzzle frames could be extracted from Video A: {video_a_path}")
    if not frames_b:
        raise ValueError(f"No muzzle frames could be extracted from Video B: {video_b_path}")

    if log_callback:
        log_callback(f"Executing 5x5 biometric all-pairs SIFT/RANSAC and Minutiae matching...")

    match_result = match_muzzle_frame_sets(
        frames_a=frames_a,
        frames_b=frames_b,
        output_dir=out_dir,
    )

    match_result["video_a"] = {
        "name": Path(video_a_path).name,
        "path": str(video_a_path),
        "frames": frames_a,
        "median_parameters": res_a.get("median_parameters", {}),
        "vector_array": res_a.get("vector_array", []),
    }
    match_result["video_b"] = {
        "name": Path(video_b_path).name,
        "path": str(video_b_path),
        "frames": frames_b,
        "median_parameters": res_b.get("median_parameters", {}),
        "vector_array": res_b.get("vector_array", []),
    }

    return match_result


def compare_two_frames_biometric(
    frame_a: Any,
    frame_b: Any,
    weight_a: float = 0.65,
    weight_b: float = 0.35,
) -> Dict[str, Any]:
    """
    Direct biometric comparison between two individual muzzle frames or crops:
    Method A: SIFT Keypoint & RANSAC Affine Inliers.
    Method B: Topological Minutiae Points & Bipartite Matching.
    Composite Score: 0.65 * score_a + 0.35 * score_b.
    """
    gray_a, skel_a, mins_a = _resolve_frame_data(frame_a)
    gray_b, skel_b, mins_b = _resolve_frame_data(frame_b)

    res_a = match_sift_ransac(gray_a, gray_b)
    res_b = match_minutiae(mins_a, mins_b)

    score_a = float(res_a.get("score_method_a", 0.0))
    score_b = float(res_b.get("score_method_b", 0.0))
    composite_score = round(float(weight_a * score_a + weight_b * score_b), 4)

    cfg = load_matcher_config()
    if cfg.get("calibrated") and cfg.get("threshold") is not None:
        thresh = float(cfg["threshold"])
    else:
        thresh = DEFAULT_MATCH_THRESHOLD

    is_match = bool(composite_score >= thresh)

    return {
        "sift_inlier_count": int(res_a.get("inlier_count", 0)),
        "sift_inlier_ratio": round(float(res_a.get("inlier_ratio", 0.0)), 4),
        "minutiae_match_fraction": round(float(res_b.get("matched_fraction", 0.0)), 4),
        "composite_score": composite_score,
        "score_method_a": round(score_a, 4),
        "score_method_b": round(score_b, 4),
        "is_match": is_match,
        "threshold": thresh,
        "method_a": res_a,
        "method_b": res_b,
    }



def extract_best_muzzles_from_video(
    pipeline,
    video_path: str,
    output_dir: str,
    video_tag: str = "A",
    frame_interval: int = 10,
    max_frames: int = 8,
    top_k: int = 3,
    json_stream: bool = False,
    fast_mode: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extracts frames from video, detects muzzles, and ranks them by sharpness & confidence.
    Returns the top_k best muzzle detections.
    Supports fast motion-blur pre-filtering and smart early exit for fast CPU processing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    sampled_count = 0
    all_crops = []

    if json_stream:
        msg = f"[INFO] Video {video_tag}: Sampling frames at interval {frame_interval} (Fast Mode: {fast_mode})..."
        print(f"[JSON_EVENT] {json.dumps({'type': 'matcher_log', 'text': msg})}", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 1. Motion blur pre-filter (skip heavily blurred frames in 0.5ms)
            if fast_mode:
                gray_p = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_var = cv2.Laplacian(gray_p, cv2.CV_64F).var()
                if blur_var < 20.0:
                    frame_idx += 1
                    continue

            res = pipeline.process_image(
                frame,
                padding_pct=0.08,
                make_square=True,
                target_size=(224, 224),
            )
            res["source"] = f"vid_{video_tag}_f{frame_idx:04d}"

            if len(res["detections"]) > 0:
                saved = pipeline.save_results(res, output_dir, save_annotated=True)
                for d in res["detections"]:
                    crop_bgr = d["cropped_bgr"]
                    q = compute_image_quality(crop_bgr, d["confidence"])
                    crop_mask = None
                    if d.get("mask") is not None and "bbox_cropped" in d:
                        cy1, cx1, cy2, cx2 = d["bbox_cropped"]
                        raw_mask = d["mask"]
                        m_crop = raw_mask[cy1:cy2, cx1:cx2]
                        if m_crop.size > 0:
                            m_u8 = (m_crop.astype(np.uint8) * 255)
                            h_c, w_c = crop_bgr.shape[:2]
                            crop_mask = cv2.resize(m_u8, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
                    item = {
                        "video_tag": video_tag,
                        "frame_idx": frame_idx,
                        "confidence": float(d["confidence"]),
                        "crop_path": d.get("saved_crop_path"),
                        "annotated_path": res.get("saved_annotated_path"),
                        "cropped_bgr": crop_bgr,
                        "mask": crop_mask,
                        "quality": q,
                    }
                    all_crops.append(item)

            sampled_count += 1

            # 2. Smart Early Exit: stop immediately once a pristine muzzle is acquired
            if fast_mode and len(all_crops) >= 1:
                pristine = [c for c in all_crops if c["confidence"] >= 0.85]
                if len(pristine) >= 1:
                    if json_stream:
                        msg = f"[Turbo Speed] Video {video_tag}: Acquired pristine muzzle crop at Frame #{frame_idx} (Conf: {pristine[0]['confidence']*100:.1f}%). Early-exit triggered for maximum speed!"
                        print(f"[JSON_EVENT] {json.dumps({'type': 'matcher_log', 'text': msg})}", flush=True)
                    break

            if max_frames and sampled_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    # Safety Fallback: if no crops found, rewind and check early frames directly
    if not all_crops:
        cap_fb = cv2.VideoCapture(video_path)
        fb_idx = 0
        while cap_fb.isOpened() and fb_idx < 30:
            ret, frame = cap_fb.read()
            if not ret:
                break
            if fb_idx % 5 == 0:
                res = pipeline.process_image(frame, padding_pct=0.08, make_square=True, target_size=(224, 224))
                if len(res["detections"]) > 0:
                    pipeline.save_results(res, output_dir, save_annotated=True)
                    for d in res["detections"]:
                        crop_bgr = d["cropped_bgr"]
                        q = compute_image_quality(crop_bgr, d["confidence"])
                        crop_mask = None
                        if d.get("mask") is not None and "bbox_cropped" in d:
                            cy1, cx1, cy2, cx2 = d["bbox_cropped"]
                            raw_mask = d["mask"]
                            m_crop = raw_mask[cy1:cy2, cx1:cx2]
                            if m_crop.size > 0:
                                m_u8 = (m_crop.astype(np.uint8) * 255)
                                h_c, w_c = crop_bgr.shape[:2]
                                crop_mask = cv2.resize(m_u8, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
                        all_crops.append({
                            "video_tag": video_tag,
                            "frame_idx": fb_idx,
                            "confidence": float(d["confidence"]),
                            "crop_path": d.get("saved_crop_path"),
                            "annotated_path": res.get("saved_annotated_path"),
                            "cropped_bgr": crop_bgr,
                            "mask": crop_mask,
                            "quality": q,
                        })
                    break
            fb_idx += 1
        cap_fb.release()

    # Sort crops by composite quality score descending
    all_crops.sort(key=lambda x: x["quality"]["composite"], reverse=True)
    best_crops = all_crops[:top_k]

    if json_stream:
        msg = f"[INFO] Video {video_tag}: Extracted {len(all_crops)} muzzles. Top quality score: {best_crops[0]['quality']['composite'] if best_crops else 0}"
        print(f"[JSON_EVENT] {json.dumps({'type': 'matcher_log', 'text': msg})}", flush=True)

    return best_crops


def execute_two_video_comparison(
    weights_path: str,
    video_a_path: str,
    video_b_path: str,
    output_dir: str,
    device: str = "gpu",
    cpu_cores: Optional[int] = None,
    upscale_method: str = "lanczos4",
    scale_factor: int = 2,
    threshold: int = DEFAULT_DUPLICATE_THRESHOLD,
    frame_interval: int = 10,
    max_frames: int = 8,
    json_stream: bool = True,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """
    Complete end-to-end comparison between two cattle videos:
    1. Loads Mask R-CNN detector with weights (640px fast mode for 2-core CPU)
    2. Extracts best quality muzzles from Video A (motion blur filter + smart early exit)
    3. Extracts best quality muzzles from Video B (motion blur filter + smart early exit)
    4. Upscales and enhances both into high-contrast Grayscale ridge prints
    5. Performs ORB keypoint matching (statusprint.py)
    6. Outputs match verdict and visualization diagram
    """
    from muzzle_cropper import MuzzleDetectionPipeline

    start_time = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    if cpu_cores is not None:
        cpu_cores = int(cpu_cores)
        if cpu_cores < 1:
            raise ValueError(f"cpu_cores must be an integer >= 1, got {cpu_cores}")
        try:
            cv2.setNumThreads(cpu_cores)
        except Exception:
            pass

    def emit(log_text):
        if json_stream:
            print(f"[JSON_EVENT] {json.dumps({'type': 'matcher_log', 'text': log_text})}", flush=True)

    emit(f"=== Starting Two-Video Muzzle Comparison ===")
    emit(f"Video A: {Path(video_a_path).name} | Video B: {Path(video_b_path).name}")
    emit(f"Device: {device.upper()} (Cores: {cpu_cores or 'Kernel Default 12 Threads'}) | Fast Mode: {fast_mode} | Upscale: {upscale_method.upper()} ({scale_factor}x) | Threshold: {threshold} matches")

    # Initialize Pipeline
    pipeline = MuzzleDetectionPipeline(
        weights_path=weights_path,
        device=device,
        cpu_cores=cpu_cores,
        confidence_threshold=0.70,
        fast_mode=fast_mode,
    )

    # 1. Extract from Video A
    emit(f"[Step 1/4] Detecting cattle muzzles from Video A...")
    dir_a = os.path.join(output_dir, "video_A_crops")
    crops_a = extract_best_muzzles_from_video(
        pipeline=pipeline,
        video_path=video_a_path,
        output_dir=dir_a,
        video_tag="A",
        frame_interval=frame_interval,
        max_frames=max_frames,
        top_k=3,
        json_stream=json_stream,
        fast_mode=fast_mode,
    )

    if not crops_a:
        raise ValueError(f"No cattle muzzle detected in Video A: {video_a_path}")

    best_a = crops_a[0]
    emit(f"Selected Best Muzzle from Video A: Frame #{best_a['frame_idx']} (Conf: {best_a['confidence']*100:.1f}%, Quality: {best_a['quality']['composite']})")

    # 2. Extract from Video B
    emit(f"[Step 2/4] Detecting cattle muzzles from Video B...")
    dir_b = os.path.join(output_dir, "video_B_crops")
    crops_b = extract_best_muzzles_from_video(
        pipeline=pipeline,
        video_path=video_b_path,
        output_dir=dir_b,
        video_tag="B",
        frame_interval=frame_interval,
        max_frames=max_frames,
        top_k=3,
        json_stream=json_stream,
        fast_mode=fast_mode,
    )

    if not crops_b:
        raise ValueError(f"No cattle muzzle detected in Video B: {video_b_path}")

    best_b = crops_b[0]
    emit(f"Selected Best Muzzle from Video B: Frame #{best_b['frame_idx']} (Conf: {best_b['confidence']*100:.1f}%, Quality: {best_b['quality']['composite']})")

    # 3. Super-Resolution & Grayscale Ridge Enhancement
    emit(f"[Step 3/4] Applying {scale_factor}x {upscale_method.upper()} upscaling & CLAHE grayscale enhancement...")
    gray_a, up_bgr_a = enhance_muzzle_image(
        best_a["cropped_bgr"],
        method=upscale_method,
        scale_factor=scale_factor,
        unsharp=True,
        clahe=True,
    )
    gray_b, up_bgr_b = enhance_muzzle_image(
        best_b["cropped_bgr"],
        method=upscale_method,
        scale_factor=scale_factor,
        unsharp=True,
        clahe=True,
    )

    # Save enhanced prints
    enhanced_a_path = os.path.join(output_dir, "enhanced_muzzle_A.jpg")
    enhanced_b_path = os.path.join(output_dir, "enhanced_muzzle_B.jpg")
    cv2.imwrite(enhanced_a_path, gray_a)
    cv2.imwrite(enhanced_b_path, gray_b)

    # 3b. Extract Muzzle Groove Ridge Networks & Topological Skeletons
    emit(f"[Step 3b/4] Extracting fine dermatoglyphic groove patterns & topological skeletons...")
    groove_res_a = extract_muzzle_grooves_pipeline(
        image_bgr=best_a["cropped_bgr"],
        mask=best_a.get("mask"),
        method="blackhat",
        scales=[9, 15, 21],
        threshold_method="otsu",
        min_component_size=15,
        exclude_nostrils=True,
        output_dir=None,
    )
    p_resp_a = os.path.join(output_dir, "groove_response_A.jpg")
    p_mask_a = os.path.join(output_dir, "groove_mask_A.png")
    p_skel_a = os.path.join(output_dir, "groove_skeleton_A.png")
    p_over_a = os.path.join(output_dir, "groove_overlay_A.jpg")
    cv2.imwrite(p_resp_a, groove_res_a["enhanced_response"])
    cv2.imwrite(p_mask_a, groove_res_a["groove_mask"])
    cv2.imwrite(p_skel_a, groove_res_a["groove_skeleton"])
    cv2.imwrite(p_over_a, groove_res_a["groove_overlay"])

    groove_res_b = extract_muzzle_grooves_pipeline(
        image_bgr=best_b["cropped_bgr"],
        mask=best_b.get("mask"),
        method="blackhat",
        scales=[9, 15, 21],
        threshold_method="otsu",
        min_component_size=15,
        exclude_nostrils=True,
        output_dir=None,
    )
    p_resp_b = os.path.join(output_dir, "groove_response_B.jpg")
    p_mask_b = os.path.join(output_dir, "groove_mask_B.png")
    p_skel_b = os.path.join(output_dir, "groove_skeleton_B.png")
    p_over_b = os.path.join(output_dir, "groove_overlay_B.jpg")
    cv2.imwrite(p_resp_b, groove_res_b["enhanced_response"])
    cv2.imwrite(p_mask_b, groove_res_b["groove_mask"])
    cv2.imwrite(p_skel_b, groove_res_b["groove_skeleton"])
    cv2.imwrite(p_over_b, groove_res_b["groove_overlay"])

    emit(
        f"Extracted groove skeletons: Video A ({groove_res_a['num_skeleton_pixels']} px, {groove_res_a['timing_ms']:.1f}ms) "
        f"| Video B ({groove_res_b['num_skeleton_pixels']} px, {groove_res_b['timing_ms']:.1f}ms)"
    )

    # 4. ORB Matching & Statusprint Decision
    emit(f"[Step 4/4] Running statusprint.py ORB feature extraction & Hamming distance matcher...")
    visual_diagram_path = os.path.join(output_dir, "muzzle_match_diagram.jpg")

    match_result = compare_two_muzzles(
        gray1=gray_a,
        gray2=gray_b,
        nfeatures=1000,
        threshold=threshold,
        bgr1=up_bgr_a,
        bgr2=up_bgr_b,
        save_visual_path=visual_diagram_path,
    )

    duration = round((time.perf_counter() - start_time), 2)
    emit(f"Comparison Finished in {duration}s!")
    emit(f"RESULT: {match_result['verdict']} | Matches: {match_result['good_matches']} / Threshold: {threshold} | Similarity: {match_result['similarity_percent']}%")

    final_payload = {
        "video_a": {
            "name": Path(video_a_path).name,
            "frame_idx": best_a["frame_idx"],
            "confidence": best_a["confidence"],
            "quality": best_a["quality"],
            "crop_path": best_a["crop_path"],
            "annotated_path": best_a["annotated_path"],
            "enhanced_path": enhanced_a_path,
            "groove_response_path": p_resp_a,
            "groove_mask_path": p_mask_a,
            "groove_skeleton_path": p_skel_a,
            "groove_overlay_path": p_over_a,
            "groove_response_url": to_output_url(p_resp_a),
            "groove_mask_url": to_output_url(p_mask_a),
            "groove_skeleton_url": to_output_url(p_skel_a),
            "groove_overlay_url": to_output_url(p_over_a),
            "groove_timings": {
                "total_ms": round(groove_res_a["timing_ms"], 2),
                "groove_pixels": groove_res_a["num_groove_pixels"],
                "skeleton_pixels": groove_res_a["num_skeleton_pixels"],
            },
        },
        "video_b": {
            "name": Path(video_b_path).name,
            "frame_idx": best_b["frame_idx"],
            "confidence": best_b["confidence"],
            "quality": best_b["quality"],
            "crop_path": best_b["crop_path"],
            "annotated_path": best_b["annotated_path"],
            "enhanced_path": enhanced_b_path,
            "groove_response_path": p_resp_b,
            "groove_mask_path": p_mask_b,
            "groove_skeleton_path": p_skel_b,
            "groove_overlay_path": p_over_b,
            "groove_response_url": to_output_url(p_resp_b),
            "groove_mask_url": to_output_url(p_mask_b),
            "groove_skeleton_url": to_output_url(p_skel_b),
            "groove_overlay_url": to_output_url(p_over_b),
            "groove_timings": {
                "total_ms": round(groove_res_b["timing_ms"], 2),
                "groove_pixels": groove_res_b["num_groove_pixels"],
                "skeleton_pixels": groove_res_b["num_skeleton_pixels"],
            },
        },
        "match": match_result,
        "upscale_method": upscale_method,
        "scale_factor": scale_factor,
        "duration_seconds": duration,
    }

    if json_stream:
        print(f"[JSON_EVENT] {json.dumps({'type': 'two_video_result', 'result': final_payload})}", flush=True)

    return final_payload


def check_positive_int(value: str) -> int:
    try:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"Invalid cpu_cores: {value}. Must be an integer >= 1.")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid cpu_cores: {value}. Must be an integer >= 1.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Two-Video Cattle Muzzle Matcher using statusprint.py")
    parser.add_argument("--weights", default="/home/adis_mayur/F_CODES/crop11dec.h5")
    parser.add_argument("--video-a", required=True, help="Path to Video A")
    parser.add_argument("--video-b", required=True, help="Path to Video B")
    parser.add_argument("--output-dir", default="/home/adis_mayur/F_CODES/Mask_RCNN/output/two_video_compare")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--cpu-cores",
        type=check_positive_int,
        default=None,
        help="Number of CPU cores/threads to allocate (any integer >= 1, e.g. 1, 2, 3, 4, 5, 6, 8, 12). Default: kernel default (12 threads).",
    )
    parser.add_argument("--upscale", default="lanczos4", choices=["lanczos4", "bicubic", "bilinear"])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True, help="Enable 640px + early exit for fast CPU processing")
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Disable fast mode (full 1024px)")
    parser.add_argument("--json-stream", action="store_true")

    args = parser.parse_args()

    execute_two_video_comparison(
        weights_path=args.weights,
        video_a_path=args.video_a,
        video_b_path=args.video_b,
        output_dir=args.output_dir,
        device=args.device,
        cpu_cores=args.cpu_cores,
        upscale_method=args.upscale,
        scale_factor=args.scale,
        threshold=args.threshold,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        json_stream=args.json_stream,
        fast_mode=args.fast_mode,
    )


# ---------------------------------------------------------------------------
# Spatial matcher API (kept alongside legacy functions for backwards compatibility)
# ---------------------------------------------------------------------------
import base64 as _spatial_base64

SPATIAL_PARAMS_PATH = SCRIPT_DIR / "config" / "matching_params.json"
SPATIAL_THRESHOLD_PATH = SCRIPT_DIR / "config" / "matching_threshold.json"


def _spatial_load_params(path="config/matching_params.json"):
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    with open(p, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _spatial_gray(frame, params=None):
    params = params or _spatial_load_params()
    image = frame.get("clahe_gray")
    if image is None:
        image = frame.get("enhanced_gray")
    if image is None:
        image = frame.get("enhanced_path") or frame.get("skeleton_path")
        if isinstance(image, str):
            p = Path(image)
            if not p.is_absolute():
                p = SCRIPT_DIR / image.lstrip("/")
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    if image is None:
        raise ValueError("Frame must contain clahe_gray, enhanced_gray, or enhanced_path.")
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = int(params.get("canonical_size", 448))
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


def _spatial_mask(frame, shape, params):
    raw = frame.get("mask")
    if raw is None:
        raw = frame.get("aligned_mask")
    if raw is None:
        raw = frame.get("muzzle_mask")
    if raw is None:
        raw = np.full(shape, 255, dtype=np.uint8)
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if raw.shape[:2] != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        raw = np.where(raw > 0, 255, 0).astype(np.uint8)
    erosion = int(params["mask_erosion_px"])
    if erosion > 0 and np.count_nonzero(raw) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
        eroded = cv2.erode(raw, kernel)
        if np.count_nonzero(eroded) >= 64:
            raw = eroded
    return raw


def _spatial_skeleton(frame, mask):
    raw = frame.get("skeleton")
    if raw is None:
        raw = frame.get("groove_skeleton")
    if raw is None:
        raw = frame.get("skeleton_path")
    if isinstance(raw, str):
        p = Path(raw)
        if not p.is_absolute():
            p = SCRIPT_DIR / raw.lstrip("/")
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.shape[:2] != mask.shape:
        raw = cv2.resize(raw, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where((raw > 0) & (mask > 0), 255, 0).astype(np.uint8)


def _spatial_minutiae(frame, skeleton, params):
    raw = frame.get("minutiae")
    if not isinstance(raw, list) or not raw:
        if skeleton is None:
            return []
        raw = extract_minutiae_points(skeleton, max_minutiae=int(params["max_minutiae_points"]))
    out = []
    for item in raw[: int(params["max_minutiae_points"])]:
        try:
            out.append({
                "x": float(item["x"]), "y": float(item["y"]),
                "type": str(item.get("type", "endpoint")),
                "angle": float(item.get("angle", item.get("orientation", 0.0))) % 360.0,
                "weight": float(item.get("weight", 1.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _spatial_patch(gray, x, y, size=16):
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half > gray.shape[1] or yi + half > gray.shape[0]:
        return None
    patch = gray[yi-half:yi+half, xi-half:xi+half].astype(np.float32)
    patch -= patch.mean()
    norm = float(np.linalg.norm(patch))
    return patch / norm if norm > 1e-6 else None


def _spatial_angle_delta(a, b):
    delta = abs((float(a) - float(b)) % 360.0)
    return min(delta, 360.0 - delta)


def _spatial_features(gray, mask, params):
    glare_mask = mask.copy()
    glare_mask[gray >= int(params["glare_threshold"])] = 0
    try:
        detector = cv2.SIFT_create(nfeatures=1800)
        method = "sift"
    except Exception:
        detector = cv2.ORB_create(nfeatures=1800)
        method = "orb"
    keypoints, descriptors = detector.detectAndCompute(gray, glare_mask)
    return method, keypoints or [], descriptors


def _spatial_method_a(gray_a, gray_b, mask_a, mask_b, params):
    method_a, kp_a, desc_a = _spatial_features(gray_a, mask_a, params)
    method_b, kp_b, desc_b = _spatial_features(gray_b, mask_b, params)
    detector_name = method_a if method_a == method_b else "mixed"
    result = {"detector": detector_name, "inlier_count": 0, "inlier_ratio": 0.0,
              "score": 0.0, "kp_a": kp_a, "kp_b": kp_b, "matches": [],
              "inlier_mask": [], "transform_matrix": None}
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return result
    norm = cv2.NORM_HAMMING if "orb" in (method_a, method_b) else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    try:
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return result
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < float(params["sift_lowe_ratio"]) * p[1].distance]
    result["matches"] = good
    if len(good) < 3:
        return result
    points_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    transform, inliers_mask = cv2.estimateAffinePartial2D(
        points_a, points_b, method=cv2.RANSAC,
        ransacReprojThreshold=float(params["sift_ransac_reproj_thresh"]),
    )
    if transform is None or inliers_mask is None:
        return result
    scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    angle = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
    if not (0.65 <= scale <= 1.55 and abs(angle) <= 55.0):
        return result
    inlier_flags = inliers_mask.ravel().astype(bool)
    inlier_count = int(inlier_flags.sum())
    inlier_ratio = inlier_count / max(1, len(good))
    score = inlier_ratio * min(1.0, inlier_count / max(1, int(params["sift_min_inliers_full_weight"])))
    result.update({"inlier_count": inlier_count, "inlier_ratio": float(inlier_ratio),
                   "score": float(np.clip(score, 0.0, 1.0)),
                   "inlier_mask": inlier_flags.tolist(), "transform_matrix": transform})
    return result


def _spatial_method_b(gray_a, gray_b, points_a, points_b, params):
    empty = {"matched_count": 0, "matched_fraction": 0.0, "score": 0.0,
             "pairs": [], "transform_matrix": None}
    if len(points_a) < 2 or len(points_b) < 2:
        return empty
    # Build plausible pairs using position, orientation, and local normalized patch correlation.
    # Endpoint/junction types are intentionally not a hard constraint.
    choices = []
    radius = float(params["minutiae_max_radius_px"])
    max_angle = float(params["minutiae_max_angle_deg"])
    min_corr = float(params["minutiae_min_patch_corr"])
    patches_a = [_spatial_patch(gray_a, p["x"], p["y"]) for p in points_a]
    patches_b = [_spatial_patch(gray_b, p["x"], p["y"]) for p in points_b]
    for i, a in enumerate(points_a):
        pa = patches_a[i]
        if pa is None:
            continue
        for j, b in enumerate(points_b):
            if _spatial_angle_delta(a["angle"], b["angle"]) > max_angle:
                continue
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > radius:
                continue
            pb = patches_b[j]
            if pb is None:
                continue
            corr = float(np.sum(pa * pb))
            if corr >= min_corr:
                choices.append((corr, i, j))
    choices.sort(reverse=True)
    candidate_pairs, used_a, used_b = [], set(), set()
    for corr, i, j in choices:
        if i not in used_a and j not in used_b:
            candidate_pairs.append((i, j, corr))
            used_a.add(i); used_b.add(j)
    if len(candidate_pairs) < 3:
        return empty
    coords_a = np.float32([[points_a[i]["x"], points_a[i]["y"]] for i, _, _ in candidate_pairs]).reshape(-1, 1, 2)
    coords_b = np.float32([[points_b[j]["x"], points_b[j]["y"]] for _, j, _ in candidate_pairs]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        coords_a, coords_b, method=cv2.RANSAC,
        ransacReprojThreshold=float(params["minutiae_ransac_reproj_thresh"]),
    )
    if transform is None or inlier_mask is None:
        return empty
    flags = inlier_mask.ravel().astype(bool)
    matched = int(flags.sum())
    fraction = matched / max(1, min(len(points_a), len(points_b)))
    return {"matched_count": matched, "matched_fraction": float(np.clip(fraction, 0.0, 1.0)),
            "score": float(np.clip(fraction, 0.0, 1.0)),
            "pairs": [candidate_pairs[i] for i, ok in enumerate(flags) if ok],
            "transform_matrix": transform}


def match_frame_pair(frame_a, frame_b, params_path="config/matching_params.json") -> Dict[str, Any]:
    params = _spatial_load_params(params_path)
    gray_a, gray_b = _spatial_gray(frame_a, params), _spatial_gray(frame_b, params)
    mask_a, mask_b = _spatial_mask(frame_a, gray_a.shape, params), _spatial_mask(frame_b, gray_b.shape, params)
    skel_a, skel_b = _spatial_skeleton(frame_a, mask_a), _spatial_skeleton(frame_b, mask_b)
    mins_a = _spatial_minutiae(frame_a, skel_a, params)
    mins_b = _spatial_minutiae(frame_b, skel_b, params)
    method_a = _spatial_method_a(gray_a, gray_b, mask_a, mask_b, params)
    method_b = _spatial_method_b(gray_a, gray_b, mins_a, mins_b, params)
    score = float(params["weight_sift"]) * method_a["score"] + float(params["weight_minutiae"]) * method_b["score"]
    return {"pair_score": float(np.clip(score, 0.0, 1.0)),
            "method_a_sift": {k: method_a[k] for k in ("detector", "inlier_count", "inlier_ratio", "score")},
            "method_b_minutiae": {k: method_b[k] for k in ("matched_count", "matched_fraction", "score")},
            "_method_a": method_a, "_method_b": method_b, "_gray_a": gray_a, "_gray_b": gray_b}


def _spatial_read_threshold(threshold_path="config/matching_threshold.json"):
    p = Path(threshold_path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        threshold = float(data["threshold"])
        genuine = int(data.get("genuine_pairs", 0))
        impostor = int(data.get("impostor_pairs", 0))
        calibrated = bool(data.get("calibrated")) and genuine >= 30 and impostor >= 30 and math.isfinite(threshold)
        return calibrated, threshold if calibrated else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, None


def _spatial_visualization(best, verdict, dev_threshold):
    method_a = best["_method_a"]
    gray_a, gray_b = best["_gray_a"], best["_gray_b"]
    good = method_a.get("matches", [])
    inlier_flags = method_a.get("inlier_mask", [])
    inlier_matches = [m for m, is_inlier in zip(good, inlier_flags) if is_inlier]
    canvas = cv2.drawMatches(cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR), method_a.get("kp_a", []),
                             cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR), method_a.get("kp_b", []),
                             inlier_matches[:80], None,
                             matchColor=(70, 220, 100) if verdict == "MATCH" else (30, 180, 255),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    banner = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f"{verdict} | spatial score {best['pair_score']:.3f} | DEV threshold {dev_threshold:.2f}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", np.vstack([banner, canvas]))
    return _spatial_base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""


def match_video_sets(frames_a, frames_b, params_path="config/matching_params.json",
                     threshold_path="config/matching_threshold.json") -> Dict[str, Any]:
    params = _spatial_load_params(params_path)
    if not frames_a or not frames_b:
        raise ValueError("Both video frame sets must contain at least one frame.")
    results = [match_frame_pair(a, b, params_path=params_path) for a in frames_a for b in frames_b]
    scores = [r["pair_score"] for r in results]
    best = results[int(np.argmax(scores))]
    best_score, median_score = float(max(scores)), float(np.median(scores))
    decision_score = float(params["weight_best_pair"]) * best_score + float(params["weight_median_pair"]) * median_score
    calibrated, threshold = _spatial_read_threshold(threshold_path)
    dev_threshold = float(params["dev_provisional_threshold"])
    verdict = ("MATCH" if decision_score >= threshold else "NO_MATCH") if calibrated and threshold is not None else "UNCALIBRATED"
    dev_hint = "DEV_LIKELY_MATCH" if decision_score >= dev_threshold else "DEV_LIKELY_NO_MATCH"
    viz = _spatial_visualization(best, verdict if calibrated else dev_hint, dev_threshold)
    return {"decision_score": round(decision_score, 5), "best_score": round(best_score, 5),
            "median_score": round(median_score, 5), "method_a_sift": best["method_a_sift"],
            "method_b_minutiae": best["method_b_minutiae"], "verdict": verdict,
            "dev_hint": dev_hint, "calibrated": calibrated, "threshold": threshold,
            "dev_provisional_threshold": dev_threshold, "match_viz_base64": viz,
            "pair_count": len(results)}
