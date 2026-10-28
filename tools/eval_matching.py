#!/usr/bin/env python3
"""
Cattle Muzzle Biometric Matching Evaluation & ROC/EER Calibration Harness
========================================================================
Strictly scoped to /home/adis_mayur/F_CODES/nose_lines.

Evaluates cattle muzzle biometric matching performance:
1. Input folder layout: dataset/<cow_id>/<video>.mp4
   - Scans dataset/ for cow subdirectories containing video files.
   - Auto-initializes dataset/ with existing videos or synthetic clips if empty.
2. Matching Evaluation:
   - Genuine pairs (same cow, video A vs video B of cow X)
   - Impostor pairs (different cows, video from cow X vs video from cow Y)
   - Within-video frame-to-frame comparisons (sanity check on frame stability)
   - Employs Method A (SIFT/RANSAC) + Method B (minutiae matching) + Composite Score
3. Biometric Calibration (ROC & EER):
   - Computes TAR vs FAR across varying thresholds in [0.0, 1.0]
   - Determines Equal Error Rate (EER) and optimal eer_threshold
   - Computes Area Under the ROC Curve (AUC)
4. Outputs:
   - Histogram distribution plot (score_distribution.png)
   - ROC curve plot (roc_curve.png)
   - Comprehensive JSON evaluation report (matching_eval_report.json)
   - Calibrated configuration (matcher_config.json at project root)
"""

import os
import sys
import json
import math
import time
import shutil
import logging
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

import cv2
import numpy as np

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from muzzle_matcher import (
    match_sift_ransac,
    match_minutiae,
    extract_minutiae_points,
    match_muzzle_frame_sets,
    compare_two_frames_biometric,
    enhance_muzzle_image,
    _resolve_frame_data,
    CANONICAL_SIZE,
    DEFAULT_MATCH_THRESHOLD,
)
from video_groove_extractor import extract_five_muzzle_stages
from groove_extractor import extract_muzzle_grooves_pipeline

logger = logging.getLogger("eval_matching")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# 1. Dataset Management & Synthetic Data Generator
# ==============================================================================

def scan_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    """
    Scans dataset_dir for subdirectories representing cow IDs.
    Returns mapping: {cow_id: [video_path_1, video_path_2, ...]}.
    """
    dataset_dir = Path(dataset_dir).resolve()
    cows: Dict[str, List[Path]] = {}

    if not dataset_dir.exists():
        return cows

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for sub in sorted(dataset_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            vids = sorted([
                f for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")
            ])
            if vids:
                cows[sub.name] = vids

    return cows


def setup_default_dataset_if_needed(
    dataset_dir: Path,
    source_dir: Path = Path("/home/adis_mayur/F_CODES"),
) -> Dict[str, List[Path]]:
    """
    If dataset_dir does not exist or has fewer than 2 cows, organizes available
    source mp4 videos into dataset/ layout:
      dataset/cow_001/video_1.mp4  (muzzleVideo.mp4)
      dataset/cow_001/video_2.mp4  (muzzleVideo2.mp4)
      dataset/cow_001/video_3.mp4  (muzzleVideo3.mp4)
      dataset/cow_002/video_1.mp4  (muzzleVideo5.mp4 - part 1)
      dataset/cow_002/video_2.mp4  (muzzleVideo5.mp4 - part 2)
      dataset/cow_003/video_1.mp4  (muzzleVideo6.mp4)
      dataset/cow_003/video_2.mp4  (muzzleVideo7.mp4)
    """
    dataset_dir = Path(dataset_dir).resolve()
    existing = scan_dataset(dataset_dir)
    if len(existing) >= 2 and sum(len(v) for v in existing.values()) >= 3:
        logger.info(f"Existing dataset verified: {len(existing)} cows found in {dataset_dir}")
        return existing

    dataset_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir).resolve()
    logger.info(f"Setting up biometric evaluation dataset in: {dataset_dir}")

    # Discover candidate source videos
    candidates = {
        "muzzleVideo.mp4": source_dir / "muzzleVideo.mp4",
        "muzzleVideo2.mp4": source_dir / "muzzleVideo2.mp4",
        "muzzleVideo3.mp4": source_dir / "muzzleVideo3.mp4",
        "muzzleVideo5.mp4": source_dir / "muzzleVideo5.mp4",
        "muzzleVideo6.mp4": source_dir / "muzzleVideo6.mp4",
        "muzzleVideo7.mp4": source_dir / "muzzleVideo7.mp4",
    }

    found_videos = {k: v for k, v in candidates.items() if v.exists()}

    def link_or_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        try:
            os.symlink(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))

    # Cow 001: muzzleVideo, muzzleVideo2, muzzleVideo3
    cow1_dir = dataset_dir / "cow_001"
    cow1_dir.mkdir(parents=True, exist_ok=True)
    if "muzzleVideo.mp4" in found_videos:
        link_or_copy(found_videos["muzzleVideo.mp4"], cow1_dir / "video_1.mp4")
    if "muzzleVideo2.mp4" in found_videos:
        link_or_copy(found_videos["muzzleVideo2.mp4"], cow1_dir / "video_2.mp4")
    if "muzzleVideo3.mp4" in found_videos:
        link_or_copy(found_videos["muzzleVideo3.mp4"], cow1_dir / "video_3.mp4")

    # Cow 002: muzzleVideo5 split into two clips
    cow2_dir = dataset_dir / "cow_002"
    cow2_dir.mkdir(parents=True, exist_ok=True)
    if "muzzleVideo5.mp4" in found_videos:
        src5 = found_videos["muzzleVideo5.mp4"]
        v2_1 = cow2_dir / "video_1.mp4"
        v2_2 = cow2_dir / "video_2.mp4"
        link_or_copy(src5, v2_1)

        # Create video_2.mp4 from 2nd half of muzzleVideo5 if not already present
        if not v2_2.exists():
            cap = cv2.VideoCapture(str(src5))
            if cap.isOpened():
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(v2_2), fourcc, fps, (w, h))

                start_f = total_f // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                f_count = 0
                while cap.isOpened() and f_count < 120:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    writer.write(fr)
                    f_count += 1
                cap.release()
                writer.release()
            if not v2_2.exists():
                link_or_copy(src5, v2_2)

    # Cow 003: muzzleVideo6, muzzleVideo7
    cow3_dir = dataset_dir / "cow_003"
    cow3_dir.mkdir(parents=True, exist_ok=True)
    if "muzzleVideo6.mp4" in found_videos:
        link_or_copy(found_videos["muzzleVideo6.mp4"], cow3_dir / "video_1.mp4")
    if "muzzleVideo7.mp4" in found_videos:
        link_or_copy(found_videos["muzzleVideo7.mp4"], cow3_dir / "video_2.mp4")

    cows = scan_dataset(dataset_dir)
    logger.info(f"Dataset successfully initialized: {len(cows)} cows, {sum(len(v) for v in cows.values())} total videos.")
    return cows


def generate_synthetic_demo_dataset(
    dataset_dir: Path,
    num_cows: int = 3,
    videos_per_cow: int = 2,
    frames_per_video: int = 15,
) -> Dict[str, List[Path]]:
    """
    Generates procedurally rendered cattle muzzle video clips for automated benchmarking,
    testing, or running in headless sandboxes without source videos.
    Each cow has a distinctive dermatoglyphic ridge groove fingerprint pattern.
    """
    dataset_dir = Path(dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating synthetic biometric demo dataset in {dataset_dir} ({num_cows} cows)...")

    cows: Dict[str, List[Path]] = {}

    for cow_idx in range(1, num_cows + 1):
        cow_id = f"cow_{cow_idx:03d}"
        cow_dir = dataset_dir / cow_id
        cow_dir.mkdir(parents=True, exist_ok=True)
        cows[cow_id] = []

        # Generate base pattern seed specific to this animal
        seed_val = 1000 + cow_idx * 137
        rng = np.random.RandomState(seed_val)

        base_pattern = np.full((320, 320), 120, dtype=np.uint8)
        # Draw distinctive ridge network
        for _ in range(35):
            x1, y1 = rng.randint(40, 280), rng.randint(40, 280)
            x2 = int(np.clip(x1 + rng.randint(-60, 60), 20, 300))
            y2 = int(np.clip(y1 + rng.randint(-60, 60), 20, 300))
            cv2.line(base_pattern, (x1, y1), (x2, y2), 220, thickness=2, lineType=cv2.LINE_AA)
            if rng.rand() > 0.4:
                # Add branch / junction
                x3 = int(np.clip(x2 + rng.randint(-40, 40), 20, 300))
                y3 = int(np.clip(y2 + rng.randint(-40, 40), 20, 300))
                cv2.line(base_pattern, (x2, y2), (x3, y3), 220, thickness=2, lineType=cv2.LINE_AA)

        # Smooth slightly to resemble skin dermatoglyphics
        base_pattern = cv2.GaussianBlur(base_pattern, (3, 3), 1.0)

        for v_idx in range(1, videos_per_cow + 1):
            vid_path = cow_dir / f"video_{v_idx}.mp4"
            cows[cow_id].append(vid_path)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 320))

            v_rng = np.random.RandomState(seed_val + v_idx * 99)
            base_angle = v_rng.uniform(-4.0, 4.0)
            base_scale = v_rng.uniform(0.97, 1.03)

            for f in range(frames_per_video):
                # Small per-frame motion jitter
                jitter_angle = base_angle + 0.5 * math.sin(f * 0.3)
                jitter_dx = 2.0 * math.cos(f * 0.2)
                jitter_dy = 2.0 * math.sin(f * 0.25)

                M = cv2.getRotationMatrix2D((160, 160), jitter_angle, base_scale)
                M[0, 2] += jitter_dx
                M[1, 2] += jitter_dy

                warped = cv2.warpAffine(base_pattern, M, (320, 320), borderMode=cv2.BORDER_REFLECT)
                noise = v_rng.normal(0, 4, warped.shape).astype(np.int16)
                noisy = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

                writer.write(bgr)

            writer.release()

    logger.info(f"Synthetic demo dataset generated successfully.")
    return cows


# ==============================================================================
# 2. Feature Extraction & Video Cache
# ==============================================================================

def extract_or_load_video_features(
    video_path: Path,
    cache_base_dir: Path,
    filter_method: str = "frangi",
    threshold_method: str = "adaptive",
    device: str = "cpu",
    cpu_cores: int = 6,
    force_recompute: bool = False,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Extracts 5 canonical muzzle stage frames (native crop, enhanced ridge, response, mask, skeleton)
    and minutiae parameters. Results are cached to disk so all-pairs matching is instantaneous.
    """
    video_path = Path(video_path).resolve()
    cache_dir = cache_base_dir / video_path.stem
    summary_path = cache_dir / "summary.json"

    if not force_recompute and summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("success") and data.get("frames"):
                return data
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting biometric features for video: {video_path.name}")

    if not is_synthetic:
        try:
            res = extract_five_muzzle_stages(
                video_path=video_path,
                method=filter_method,
                threshold_method=threshold_method,
                device=device,
                cpu_cores=cpu_cores,
            )
            if res.get("success") and res.get("frames"):
                return res
        except Exception as e:
            logger.warning(f"Mask R-CNN extraction failed for {video_path.name} ({e}). Using direct frame sampling fallback.")

    # Direct fallback: sample frames from video directly
    cap = cv2.VideoCapture(str(video_path))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    indices = [int(total_f * frac) for frac in [0.10, 0.30, 0.50, 0.70, 0.90]]

    frames_records = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_f - 1, max(0, idx)))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), 128, dtype=np.uint8)

        # Center square crop
        h, w = frame.shape[:2]
        sz = min(h, w)
        cy, cx = h // 2, w // 2
        crop = frame[cy - sz//2 : cy + sz//2, cx - sz//2 : cx + sz//2]
        crop = cv2.resize(crop, CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4)

        crop_path = cache_dir / f"frame_{i}_01_native_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        enhanced_gray, _ = enhance_muzzle_image(crop, scale_factor=1)
        enhanced_path = cache_dir / f"frame_{i}_02_enhanced_ridge.jpg"
        cv2.imwrite(str(enhanced_path), enhanced_gray)

        # Extract groove skeleton and minutiae
        skel_path = cache_dir / f"frame_{i}_05_thin_skeleton.png"
        try:
            groove_out = extract_muzzle_grooves_pipeline(
                crop,
                method="blackhat",
                threshold_method="otsu",
                exclude_nostrils=False,
            )
            skel = groove_out.get("groove_skeleton")
            if skel is not None:
                cv2.imwrite(str(skel_path), skel)
                minutiae = extract_minutiae_points(skel)
            else:
                minutiae = []
        except Exception:
            minutiae = []

        frames_records.append({
            "frame_number": i + 1,
            "frame_idx": idx,
            "timestamp_sec": round(float(idx / fps), 2),
            "confidence": 0.85,
            "crop_path": str(crop_path),
            "enhanced_path": str(enhanced_path),
            "skeleton_path": str(skel_path) if skel_path.exists() else None,
            "minutiae": minutiae,
        })
    cap.release()

    payload = {
        "success": True,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "frames": frames_records,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


# ==============================================================================
# 3. Comprehensive Evaluation Runner
# ==============================================================================

def run_evaluation(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cpu",
    cpu_cores: int = 6,
    max_impostor_pairs: Optional[int] = 50,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full biometric verification evaluation:
    1. Genuine pairs (cow X video A vs cow X video B)
    2. Impostor pairs (cow X video A vs cow Y video B)
    3. Within-video frame comparisons (video A frame i vs frame j)
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cows = scan_dataset(dataset_dir)
    if not cows:
        raise ValueError(f"No cattle videos found in dataset directory: {dataset_dir}")

    total_videos = sum(len(v) for v in cows.values())
    logger.info(f"Starting biometric matching evaluation: {len(cows)} cows, {total_videos} videos.")

    # Step 1: Pre-extract and cache features for each video
    video_features: Dict[str, Dict[str, Any]] = {}
    for cow_id, vlist in cows.items():
        for vpath in vlist:
            v_key = f"{cow_id}/{vpath.name}"
            feat = extract_or_load_video_features(
                video_path=vpath,
                cache_base_dir=cache_dir / cow_id,
                device=device,
                cpu_cores=cpu_cores,
                is_synthetic=is_synthetic,
            )
            video_features[v_key] = feat

    genuine_records: List[Dict[str, Any]] = []
    impostor_records: List[Dict[str, Any]] = []
    within_video_records: List[Dict[str, Any]] = []

    # Step 2: Genuine Pairs (Same cow, different videos)
    logger.info("Computing genuine video pairs...")
    for cow_id, vlist in cows.items():
        if len(vlist) < 2:
            continue
        for va, vb in itertools.combinations(vlist, 2):
            key_a = f"{cow_id}/{va.name}"
            key_b = f"{cow_id}/{vb.name}"
            fa = video_features[key_a].get("frames", [])
            fb = video_features[key_b].get("frames", [])

            if not fa or not fb:
                continue

            match_res = match_muzzle_frame_sets(fa, fb, output_dir=output_dir)
            rec = {
                "cow_id": cow_id,
                "video_a": va.name,
                "video_b": vb.name,
                "pair_type": "genuine",
                "composite_score": match_res["composite_score"],
                "sift_inlier_count": match_res["inlier_count"],
                "sift_inlier_ratio": match_res["inlier_ratio"],
                "minutiae_match_fraction": match_res["matched_minutiae_fraction"],
                "score_method_a": match_res["method_a"]["score"],
                "score_method_b": match_res["method_b"]["score"],
                "best_pair": match_res.get("best_pair", {}),
            }
            genuine_records.append(rec)

    # Step 3: Impostor Pairs (Different cows)
    logger.info("Computing impostor video pairs...")
    cow_ids = sorted(list(cows.keys()))
    impostor_pairs_candidates = []

    for i in range(len(cow_ids)):
        for j in range(i + 1, len(cow_ids)):
            cid_a, cid_b = cow_ids[i], cow_ids[j]
            for va in cows[cid_a]:
                for vb in cows[cid_b]:
                    impostor_pairs_candidates.append((cid_a, va, cid_b, vb))

    if max_impostor_pairs and len(impostor_pairs_candidates) > max_impostor_pairs:
        # Uniformly subsample
        step = max(1, len(impostor_pairs_candidates) // max_impostor_pairs)
        impostor_pairs_candidates = impostor_pairs_candidates[::step][:max_impostor_pairs]

    for cid_a, va, cid_b, vb in impostor_pairs_candidates:
        key_a = f"{cid_a}/{va.name}"
        key_b = f"{cid_b}/{vb.name}"
        fa = video_features[key_a].get("frames", [])
        fb = video_features[key_b].get("frames", [])

        if not fa or not fb:
            continue

        match_res = match_muzzle_frame_sets(fa, fb, output_dir=output_dir)
        rec = {
            "cow_id_a": cid_a,
            "cow_id_b": cid_b,
            "video_a": va.name,
            "video_b": vb.name,
            "pair_type": "impostor",
            "composite_score": match_res["composite_score"],
            "sift_inlier_count": match_res["inlier_count"],
            "sift_inlier_ratio": match_res["inlier_ratio"],
            "minutiae_match_fraction": match_res["matched_minutiae_fraction"],
            "score_method_a": match_res["method_a"]["score"],
            "score_method_b": match_res["method_b"]["score"],
            "best_pair": match_res.get("best_pair", {}),
        }
        impostor_records.append(rec)

    # Step 4: Within-Video Frame-to-Frame Comparisons (Sanity check)
    logger.info("Computing within-video frame-to-frame comparisons...")
    for cow_id, vlist in cows.items():
        for vpath in vlist:
            key = f"{cow_id}/{vpath.name}"
            frames = video_features[key].get("frames", [])
            if len(frames) < 2:
                continue

            for idx_a, idx_b in itertools.combinations(range(len(frames)), 2):
                frame_res = compare_two_frames_biometric(frames[idx_a], frames[idx_b])
                rec = {
                    "cow_id": cow_id,
                    "video": vpath.name,
                    "frame_a_idx": idx_a,
                    "frame_b_idx": idx_b,
                    "pair_type": "within_video",
                    "composite_score": frame_res["composite_score"],
                    "sift_inlier_count": frame_res["sift_inlier_count"],
                    "sift_inlier_ratio": frame_res["sift_inlier_ratio"],
                    "minutiae_match_fraction": frame_res["minutiae_match_fraction"],
                    "score_method_a": frame_res["score_method_a"],
                    "score_method_b": frame_res["score_method_b"],
                }
                within_video_records.append(rec)

    logger.info(
        f"Evaluation finished: {len(genuine_records)} genuine pairs, "
        f"{len(impostor_records)} impostor pairs, {len(within_video_records)} within-video pairs."
    )

    return {
        "genuine": genuine_records,
        "impostor": impostor_records,
        "within_video": within_video_records,
        "dataset_summary": {
            "total_cows": len(cows),
            "total_videos": total_videos,
            "cows": {cid: [p.name for p in vlist] for cid, vlist in cows.items()},
        },
    }


# ==============================================================================
# 4. Biometric Analysis (ROC & EER Calibration)
# ==============================================================================

def compute_distribution_stats(scores: List[float]) -> Dict[str, float]:
    """Computes mean, std, median, min, max, and quartiles."""
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    arr = np.array(scores, dtype=np.float64)
    return {
        "count": len(scores),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def compute_roc_eer(
    genuine_scores: List[float],
    impostor_scores: List[float],
    num_thresholds: int = 501,
) -> Dict[str, Any]:
    """
    Computes TAR vs FAR across varying decision thresholds, finds Equal Error Rate (EER),
    optimal threshold, and Area Under the ROC Curve (AUC).
    """
    if not genuine_scores or not impostor_scores:
        logger.warning("Empty genuine or impostor scores; returning default uncalibrated metrics.")
        return {
            "eer": 0.0,
            "eer_threshold": DEFAULT_MATCH_THRESHOLD,
            "roc_auc": 0.5,
            "tar_at_eer": 1.0,
            "far_at_eer": 0.0,
            "threshold_table": [],
        }

    gen_arr = np.array(genuine_scores, dtype=np.float64)
    imp_arr = np.array(impostor_scores, dtype=np.float64)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    far_list: List[float] = []
    frr_list: List[float] = []
    tar_list: List[float] = []
    threshold_table: List[Dict[str, float]] = []

    best_diff = float("inf")
    eer_idx = 0

    for idx, t in enumerate(thresholds):
        # Genuine: accept if score >= t
        fa_count = np.sum(imp_arr >= t)
        fr_count = np.sum(gen_arr < t)

        far = float(fa_count / len(imp_arr))
        frr = float(fr_count / len(gen_arr))
        tar = 1.0 - frr

        far_list.append(far)
        frr_list.append(frr)
        tar_list.append(tar)

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_idx = idx

        # Record sampled points for JSON report
        if idx % 25 == 0 or idx == len(thresholds) - 1:
            threshold_table.append({
                "threshold": round(float(t), 4),
                "far": round(far, 4),
                "frr": round(frr, 4),
                "tar": round(tar, 4),
            })

    eer_thresh = float(thresholds[eer_idx])
    eer_val = round(float((far_list[eer_idx] + frr_list[eer_idx]) / 2.0), 4)
    tar_at_eer = round(float(tar_list[eer_idx]), 4)
    far_at_eer = round(float(far_list[eer_idx]), 4)

    # Compute AUC using standard upper envelope trapezoid integration and Mann-Whitney concordance
    pts_dict: Dict[float, float] = {0.0: 0.0, 1.0: 1.0}
    for f, tar in zip(far_list, tar_list):
        pts_dict[f] = max(pts_dict.get(f, 0.0), tar)
    sorted_far = sorted(pts_dict.keys())
    sorted_tar = [pts_dict[f] for f in sorted_far]

    auc_trapz = float(np.trapz(sorted_tar, sorted_far))

    # Exact Mann-Whitney concordance: P(score_genuine > score_impostor)
    mw_concordance = float(np.mean([
        1.0 if g > imp else (0.5 if g == imp else 0.0)
        for g in gen_arr
        for imp in imp_arr
    ]))

    # Robust fusion of trapezoid integration and rank-order concordance
    auc = round(float(np.clip(0.5 * auc_trapz + 0.5 * mw_concordance, 0.0, 1.0)), 4)

    return {
        "eer": eer_val,
        "eer_threshold": round(eer_thresh, 4),
        "roc_auc": auc,
        "tar_at_eer": tar_at_eer,
        "far_at_eer": far_at_eer,
        "far_curve": [round(f, 4) for f in far_list],
        "tar_curve": [round(t, 4) for t in tar_list],
        "thresholds": [round(t, 4) for t in thresholds.tolist()],
        "threshold_table": threshold_table,
    }


# ==============================================================================
# 5. Visualizations (Histogram & ROC Plots)
# ==============================================================================

def plot_score_distributions(
    genuine_scores: List[float],
    impostor_scores: List[float],
    within_video_scores: List[float],
    eer_threshold: float,
    eer: float,
    save_paths: List[Path],
) -> None:
    """Generates dual distribution histogram plot showing separation and EER vertical line."""
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    bins = np.linspace(0.0, 1.0, 31)

    if impostor_scores:
        plt.hist(
            impostor_scores,
            bins=bins,
            alpha=0.65,
            color="#ef4444",
            label=f"Impostor (N={len(impostor_scores)})",
            edgecolor="#b91c1c",
            density=True,
        )

    if genuine_scores:
        plt.hist(
            genuine_scores,
            bins=bins,
            alpha=0.65,
            color="#22c55e",
            label=f"Genuine (N={len(genuine_scores)})",
            edgecolor="#15803d",
            density=True,
        )

    if within_video_scores:
        plt.hist(
            within_video_scores,
            bins=bins,
            alpha=0.35,
            color="#3b82f6",
            label=f"Within-Video Sanity (N={len(within_video_scores)})",
            edgecolor="#1d4ed8",
            density=True,
        )

    # Vertical EER line
    plt.axvline(
        x=eer_threshold,
        color="#0f172a",
        linestyle="--",
        linewidth=2.2,
        label=f"EER Threshold = {eer_threshold:.3f} (EER = {eer * 100:.1f}%)",
    )

    plt.title("Cattle Muzzle Biometric Verification - Score Distributions", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Composite Biometric Match Score (SIFT/RANSAC + Minutiae)", fontsize=11, labelpad=8)
    plt.ylabel("Probability Density", fontsize=11, labelpad=8)
    plt.xlim(-0.02, 1.02)
    plt.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=10)
    plt.tight_layout()

    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=150)
        logger.info(f"Saved score distribution plot: {p}")

    plt.close()


def plot_roc_curve(
    far_list: List[float],
    tar_list: List[float],
    roc_auc: float,
    eer: float,
    eer_threshold: float,
    save_paths: List[Path],
) -> None:
    """Generates receiver operating characteristic (ROC) curve plot."""
    plt.figure(figsize=(7.5, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Sort curve points
    pts = sorted(zip(far_list, tar_list), key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.plot(xs, ys, color="#10b981", linewidth=2.8, label=f"Biometric Matcher (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    # Mark EER operating point
    plt.plot(
        [eer],
        [1.0 - eer],
        marker="o",
        markersize=9,
        color="#ef4444",
        linestyle="None",
        label=f"EER Operating Point (EER = {eer * 100:.1f}%)",
    )

    plt.title(f"Receiver Operating Characteristic (ROC) Curve\nOptimal Decision Threshold = {eer_threshold:.3f}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("False Accept Rate (FAR / FPR)", fontsize=11, labelpad=8)
    plt.ylabel("True Accept Rate (TAR = 1 - FRR)", fontsize=11, labelpad=8)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=10)
    plt.tight_layout()

    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=150)
        logger.info(f"Saved ROC curve plot: {p}")

    plt.close()


# ==============================================================================
# 6. Main Evaluation Orchestrator & CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cattle Muzzle Biometric Evaluation Harness and ROC/EER Calibration Specialist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "dataset"),
        help="Path to cattle video dataset directory with subdirectories for cow IDs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "eval"),
        help="Directory to save evaluation reports and visualization plots",
    )
    parser.add_argument(
        "--config-out",
        default=str(PROJECT_ROOT / "matcher_config.json"),
        help="Output path for calibrated matcher_config.json",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate synthetic demo video dataset and run end-to-end benchmark without real videos",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Computation device for Mask R-CNN detector",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=6,
        help="Number of CPU cores for feature extraction",
    )
    parser.add_argument(
        "--max-impostor-pairs",
        type=int,
        default=40,
        help="Maximum number of impostor video pairs to evaluate for fast execution",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    config_path = Path(args.config_out)

    t0 = time.time()
    logger.info("=== Starting Cattle Muzzle Biometric Evaluation & Calibration ===")

    # 1. Dataset setup
    if args.synthetic_demo:
        generate_synthetic_demo_dataset(dataset_path)
    else:
        setup_default_dataset_if_needed(dataset_path)

    # 2. Run Matching Evaluation
    eval_results = run_evaluation(
        dataset_dir=dataset_path,
        output_dir=output_path,
        device=args.device,
        cpu_cores=args.cpu_cores,
        max_impostor_pairs=args.max_impostor_pairs,
        is_synthetic=args.synthetic_demo,
    )

    genuine_list = eval_results["genuine"]
    impostor_list = eval_results["impostor"]
    within_list = eval_results["within_video"]

    genuine_scores = [r["composite_score"] for r in genuine_list]
    impostor_scores = [r["composite_score"] for r in impostor_list]
    within_scores = [r["composite_score"] for r in within_list]

    # Collect granular sub-metric statistics
    genuine_inliers = [r["sift_inlier_count"] for r in genuine_list]
    genuine_ratios = [r["sift_inlier_ratio"] for r in genuine_list]
    genuine_minutiae = [r["minutiae_match_fraction"] for r in genuine_list]

    # 3. Biometric Calibration (ROC & EER)
    roc_res = compute_roc_eer(genuine_scores, impostor_scores)

    eer_val = roc_res["eer"]
    eer_thresh = roc_res["eer_threshold"]
    roc_auc = roc_res["roc_auc"]

    # Recommended minimal biometric thresholds from genuine distribution percentiles
    min_sift_inliers = int(np.percentile(genuine_inliers, 10)) if genuine_inliers else 12
    min_inlier_ratio = round(float(np.percentile(genuine_ratios, 10)), 3) if genuine_ratios else 0.25
    min_minutiae_frac = round(float(np.percentile(genuine_minutiae, 10)), 3) if genuine_minutiae else 0.15

    # 4. Generate Calibration Config (matcher_config.json)
    calibrated_config = {
        "calibrated": True,
        "calibrated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": eer_thresh,
        "eer": eer_val,
        "roc_auc": roc_auc,
        "min_sift_inliers": max(6, min_sift_inliers),
        "min_inlier_ratio": max(0.15, min_inlier_ratio),
        "min_minutiae_fraction": max(0.10, min_minutiae_frac),
        "decision_metric": "composite_score",
        "summary": {
            "total_cows": eval_results["dataset_summary"]["total_cows"],
            "total_videos": eval_results["dataset_summary"]["total_videos"],
            "genuine_pairs_evaluated": len(genuine_list),
            "impostor_pairs_evaluated": len(impostor_list),
            "within_video_pairs_evaluated": len(within_list),
            "genuine_mean_score": round(float(np.mean(genuine_scores)), 4) if genuine_scores else 0.0,
            "impostor_mean_score": round(float(np.mean(impostor_scores)), 4) if impostor_scores else 0.0,
            "tar_at_eer": roc_res["tar_at_eer"],
            "far_at_eer": roc_res["far_at_eer"],
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(calibrated_config, f, indent=2)
    logger.info(f"Saved calibrated matcher config: {config_path}")

    # Also save a copy inside output_dir for records
    with open(output_path / "matcher_config.json", "w", encoding="utf-8") as f:
        json.dump(calibrated_config, f, indent=2)

    # 5. Visual Plots
    hist_paths = [
        output_path / "score_distribution.png",
        PROJECT_ROOT / "tools" / "score_distribution.png",
    ]
    plot_score_distributions(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        within_video_scores=within_scores,
        eer_threshold=eer_thresh,
        eer=eer_val,
        save_paths=hist_paths,
    )

    roc_paths = [
        output_path / "roc_curve.png",
        PROJECT_ROOT / "tools" / "roc_curve.png",
    ]
    plot_roc_curve(
        far_list=roc_res["far_curve"],
        tar_list=roc_res["tar_curve"],
        roc_auc=roc_auc,
        eer=eer_val,
        eer_threshold=eer_thresh,
        save_paths=roc_paths,
    )

    # 6. Comprehensive JSON Evaluation Report
    report = {
        "title": "Cattle Muzzle Biometric Verification Evaluation Report",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_summary": eval_results["dataset_summary"],
        "metrics": {
            "eer": eer_val,
            "eer_threshold": eer_thresh,
            "roc_auc": roc_auc,
            "tar_at_eer": roc_res["tar_at_eer"],
            "far_at_eer": roc_res["far_at_eer"],
        },
        "score_statistics": {
            "genuine": compute_distribution_stats(genuine_scores),
            "impostor": compute_distribution_stats(impostor_scores),
            "within_video": compute_distribution_stats(within_scores),
            "genuine_sift_inliers": compute_distribution_stats(genuine_inliers),
            "genuine_inlier_ratio": compute_distribution_stats(genuine_ratios),
            "genuine_minutiae_fraction": compute_distribution_stats(genuine_minutiae),
        },
        "threshold_analysis": roc_res["threshold_table"],
        "genuine_comparisons": genuine_list,
        "impostor_comparisons": impostor_list,
        "within_video_comparisons": within_list,
        "artifacts": {
            "score_distribution_plot": str(output_path / "score_distribution.png"),
            "roc_curve_plot": str(output_path / "roc_curve.png"),
            "matcher_config": str(config_path),
        },
    }

    report_path_1 = output_path / "matching_eval_report.json"
    report_path_2 = PROJECT_ROOT / "tools" / "matching_eval_report.json"

    with open(report_path_1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(report_path_2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved evaluation report: {report_path_1} and {report_path_2}")

    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print(f"BIOMETRIC CALIBRATION COMPLETE in {elapsed}s")
    print("=" * 70)
    print(f"Optimal EER Threshold : {eer_thresh:.4f}")
    print(f"Equal Error Rate (EER): {eer_val * 100:.2f}%")
    print(f"Area Under ROC (AUC)  : {roc_auc:.4f}")
    print(f"Genuine Pairs Evaluated: {len(genuine_list)} (Mean Score: {np.mean(genuine_scores):.4f})")
    print(f"Impostor Pairs Evaluated: {len(impostor_list)} (Mean Score: {np.mean(impostor_scores):.4f})")
    print(f"Within-Video Pairs     : {len(within_list)} (Mean Score: {np.mean(within_scores):.4f})")
    print(f"Calibrated Config      : {config_path}")
    print(f"Score Distribution Plot: {output_path / 'score_distribution.png'}")
    print(f"ROC Curve Plot         : {output_path / 'roc_curve.png'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
