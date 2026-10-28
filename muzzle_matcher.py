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
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

# Ensure mrcnn is in path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from groove_extractor import extract_muzzle_grooves_pipeline


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
