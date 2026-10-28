#!/usr/bin/env python3
"""
Cattle Muzzle Video-to-Groove Extraction & Skeletonization Pipeline
===================================================================
End-to-end command-line processor:
1. Samples frames from an input cattle video with motion-blur filtering.
2. Detects cattle muzzle and segmentation mask using Mask R-CNN.
3. Ranks candidate crops by composite quality to select the pristine muzzle.
4. Executes 5-stage dermatoglyphic groove extraction:
   - Stage A: Selects native crop
   - Stage B: Isolates valid muzzle skin and suppresses background & nostrils
   - Stage C: Multi-scale morphological Black-Hat filtering (or Frangi vesselness)
   - Stage D: Produces binary groove map with connected component noise pruning
   - Stage E: Produces 1-pixel-wide topological centerline skeleton network
5. Saves all individual stage artifacts and a 5-stage visual comparison panel.

Usage:
    python process_video_grooves.py --video /home/adis_mayur/F_CODES/muzzleVideo.mp4
    python process_video_grooves.py --video /home/adis_mayur/F_CODES/muzzleVideo.mp4 --method frangi --threshold adaptive
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tabulate import tabulate

import config
config.enforce_cpu_lock()

from muzzle_cropper import MuzzleDetectionPipeline
from muzzle_matcher import compute_image_quality
from groove_extractor import extract_muzzle_grooves_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("process_video_grooves")


def create_stages_summary_panel(
    native_crop: np.ndarray,
    enhanced_ridge: np.ndarray,
    groove_resp: np.ndarray,
    groove_mask: np.ndarray,
    groove_skel: np.ndarray,
    groove_overlay: np.ndarray,
    title: str = "Muzzle Groove Line Extraction & Skeletonization"
) -> np.ndarray:
    """
    Creates a single composite image visualizing all stages side-by-side with titles.
    """
    target_h = 280
    def prep_img(im):
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        new_w = int(w * (target_h / float(h)))
        return cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)

    p1 = prep_img(native_crop)
    p2 = prep_img(enhanced_ridge)
    p3 = prep_img(groove_resp)
    p4 = prep_img(groove_mask)
    p5 = prep_img(groove_skel)
    p6 = prep_img(groove_overlay)

    panels = [
        ("1. Native Crop", p1, (200, 200, 200)),
        ("2. Lanczos-4+CLAHE", p2, (200, 200, 200)),
        ("3. Groove Response", p3, (0, 220, 255)),
        ("4. Groove Binary Mask", p4, (255, 255, 255)),
        ("5. 1px Skeleton", p5, (0, 255, 128)),
        ("6. Skeleton Overlay", p6, (0, 255, 0)),
    ]

    labeled_panels = []
    for label, panel, color in panels:
        ph, pw = panel.shape[:2]
        canvas = np.zeros((ph + 36, pw, 3), dtype=np.uint8)
        canvas[36:, :] = panel
        # Header banner
        cv2.rectangle(canvas, (0, 0), (pw, 36), (24, 24, 27), -1)
        cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 36), (pw - 1, ph + 35), (60, 60, 60), 1)
        labeled_panels.append(canvas)

    total_w = sum(p.shape[1] for p in labeled_panels)
    full_h = target_h + 36 + 48
    composite = np.zeros((full_h, total_w, 3), dtype=np.uint8)
    composite[:48, :] = (18, 18, 18)
    cv2.putText(composite, f"CATTLE BIOMETRIC: {title}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    curr_x = 0
    for p in labeled_panels:
        pw = p.shape[1]
        composite[48:48 + p.shape[0], curr_x:curr_x + pw] = p
        curr_x += pw

    return composite


def extract_best_muzzle_from_video(
    video_path: str,
    cropper: MuzzleDetectionPipeline,
    interval: int = config.FRAME_INTERVAL,
    blur_threshold: float = config.BLUR_VARIANCE_THRESHOLD,
    max_candidates: int = config.MAX_CANDIDATE_FRAMES,
) -> Dict[str, Any]:
    """
    Samples video frames, filters blur, detects cattle muzzles, and selects
    the highest quality candidate crop.
    """
    v_path = Path(video_path).resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {v_path}")

    candidate_crops = []
    frame_idx = 0
    sampled_count = 0

    logger.info(f"Scanning video: {v_path.name} (Sampling interval: {interval} frames, Blur threshold: {blur_threshold})")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            sampled_count += 1
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if blur_var >= blur_threshold:
                # Detect muzzle using Mask R-CNN
                det_res = cropper.process_image(frame_bgr, make_square=True, target_size=(224, 224))
                for det in det_res.get("detections", []):
                    conf = det["confidence"]
                    crop_bgr = det["cropped_bgr"]
                    y1, x1, y2, x2 = det["bbox_cropped"]
                    native_bgr = frame_bgr[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else crop_bgr.copy()
                    
                    full_mask = det.get("mask")
                    crop_mask = None
                    if full_mask is not None and full_mask.shape[0] >= y2 and full_mask.shape[1] >= x2:
                        raw_patch = full_mask[y1:y2, x1:x2].astype(np.uint8) * 255
                        crop_mask = cv2.resize(raw_patch, (224, 224), interpolation=cv2.INTER_NEAREST)

                    quality = compute_image_quality(crop_bgr, conf)

                    candidate_crops.append({
                        "frame_idx": frame_idx,
                        "confidence": conf,
                        "quality": quality,
                        "crop_bgr": crop_bgr,
                        "native_crop_bgr": native_bgr,
                        "crop_mask": crop_mask,
                        "bbox_cropped": det.get("bbox_cropped"),
                    })

                    # Pristine early exit
                    if conf >= config.PRISTINE_CONFIDENCE_THRESHOLD and quality["composite"] >= 50.0:
                        logger.info(f"Early exit: Found pristine muzzle at frame #{frame_idx} (Conf: {conf*100:.1f}%, Quality: {quality['composite']}/100)")
                        break

            if candidate_crops and candidate_crops[-1]["confidence"] >= config.PRISTINE_CONFIDENCE_THRESHOLD and candidate_crops[-1]["quality"]["composite"] >= 50.0:
                break

            if len(candidate_crops) >= max_candidates:
                break

        frame_idx += 1

    cap.release()

    if not candidate_crops:
        raise RuntimeError(f"No cattle muzzle detected in video: {v_path}")

    # Rank by composite quality score descending
    candidate_crops.sort(key=lambda c: c["quality"]["composite"], reverse=True)
    best_crop = candidate_crops[0]

    logger.info(
        f"Selected Best Muzzle: Frame #{best_crop['frame_idx']} | "
        f"Confidence: {best_crop['confidence']*100:.1f}% | "
        f"Quality: {best_crop['quality']['composite']}/100 | "
        f"Sharpness: {best_crop['quality']['sharpness']}"
    )
    return best_crop


def process_video_grooves(
    video_path: str,
    output_dir: Optional[str] = None,
    method: str = "blackhat",
    scales: Optional[List[int]] = None,
    threshold_method: str = "otsu",
    min_component_size: int = 15,
    exclude_nostrils: bool = True,
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to extract muzzle crops from video and run the groove extraction pipeline.
    """
    t_start = time.perf_counter()
    out_dir = Path(output_dir or config.OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v_path = Path(video_path).resolve()
    stem = v_path.stem
    video_out_dir = out_dir / f"grooves_{stem}"
    video_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Cropper
    logger.info("Initializing Mask R-CNN Detector...")
    cropper = MuzzleDetectionPipeline(
        weights_path=weights_path or config.WEIGHTS_PATH,
        device=config.DEVICE,
        cpu_cores=config.DEFAULT_CPU_CORES
    )

    # 2. Extract Best Muzzle Crop from Video
    t_extract_start = time.perf_counter()
    best_candidate = extract_best_muzzle_from_video(str(v_path), cropper)
    t_extract = (time.perf_counter() - t_extract_start) * 1000

    native_crop = best_candidate.get("native_crop_bgr", best_candidate["crop_bgr"])
    muzzle_mask = best_candidate.get("crop_mask")

    # Save native crop
    native_crop_path = video_out_dir / "01_native_muzzle_crop.jpg"
    cv2.imwrite(str(native_crop_path), native_crop)

    # 2x Grayscale ridge enhancement (Lanczos-4 + CLAHE) for comparison
    h, w = native_crop.shape[:2]
    upscaled = cv2.resize(native_crop, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    gray_up = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_ridge = clahe_filter.apply(gray_up)
    ridge_path = video_out_dir / "02_lanczos_clahe_ridge.jpg"
    cv2.imwrite(str(ridge_path), enhanced_ridge)

    # 3. Execute Groove Extraction & Skeletonization Pipeline
    t_groove_start = time.perf_counter()
    actual_scales = scales or config.GROOVE_BLACKHAT_SCALES
    groove_res = extract_muzzle_grooves_pipeline(
        image_bgr=native_crop,
        mask=muzzle_mask,
        method=method,
        scales=actual_scales,
        threshold_method=threshold_method,
        min_component_size=min_component_size,
        exclude_nostrils=exclude_nostrils,
        output_dir=str(video_out_dir),
        prefix="groove"
    )
    t_groove = (time.perf_counter() - t_groove_start) * 1000
    total_time = (time.perf_counter() - t_start) * 1000

    # 4. Generate composite 5-stage comparison panel
    summary_panel = create_stages_summary_panel(
        native_crop=native_crop,
        enhanced_ridge=enhanced_ridge,
        groove_resp=groove_res["enhanced_response"],
        groove_mask=groove_res["groove_mask"],
        groove_skel=groove_res["groove_skeleton"],
        groove_overlay=groove_res["groove_overlay"],
        title=f"{v_path.name} (Method: {method.upper()}, Thresh: {threshold_method.upper()})"
    )
    summary_path = video_out_dir / "00_groove_5stages_summary.jpg"
    cv2.imwrite(str(summary_path), summary_panel)

    # Prepare summary results
    result = {
        "video": str(v_path),
        "frame_idx": best_candidate["frame_idx"],
        "confidence": best_candidate["confidence"],
        "quality": best_candidate["quality"],
        "resolution": f"{w}x{h}",
        "method": method,
        "scales": actual_scales,
        "threshold_method": threshold_method,
        "min_component_size": min_component_size,
        "groove_pixel_count": groove_res["groove_pixel_count"],
        "skeleton_pixel_count": groove_res["skeleton_pixel_count"],
        "groove_coverage_pct": groove_res["groove_coverage_pct"],
        "timings": {
            "extract_ms": round(t_extract, 1),
            "groove_ms": round(t_groove, 1),
            "total_ms": round(total_time, 1),
        },
        "artifacts": {
            "summary_panel": str(summary_path),
            "native_crop": str(native_crop_path),
            "enhanced_ridge": str(ridge_path),
            "groove_response": groove_res["saved_paths"]["groove_response"],
            "groove_mask": groove_res["saved_paths"]["groove_mask"],
            "groove_skeleton": groove_res["saved_paths"]["groove_skeleton"],
            "groove_overlay": groove_res["saved_paths"]["groove_overlay"],
        }
    }

    # Print summary table
    print("\n" + "=" * 75)
    print(" CATTLE MUZZLE GROOVE EXTRACTION & SKELETONIZATION RESULTS")
    print("=" * 75)
    table_data = [
        ["Input Video", str(v_path.name)],
        ["Best Muzzle Frame", f"#{best_candidate['frame_idx']} (Confidence: {best_candidate['confidence']*100:.1f}%)"],
        ["Crop Quality / Sharpness", f"{best_candidate['quality']['composite']}/100 | {best_candidate['quality']['sharpness']} var"],
        ["Input Resolution", f"{w}x{h} px"],
        ["Enhancement Method", method.upper()],
        ["Scales / Kernels", str(actual_scales)],
        ["Thresholding Method", threshold_method.upper()],
        ["Min Component Size", f"{min_component_size} px"],
        ["Groove Pixels", f"{groove_res['groove_pixel_count']:,} px ({groove_res['groove_coverage_pct']}%)"],
        ["Skeleton Pixels (1px)", f"{groove_res['skeleton_pixel_count']:,} px"],
        ["Video Extract Time", f"{t_extract:.1f} ms"],
        ["Groove Pipeline Time", f"{t_groove:.1f} ms"],
        ["Total Processing Time", f"{total_time:.1f} ms ({1000.0/total_time:.1f} FPS)"],
        ["Composite Visual Panel", str(summary_path)],
    ]
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print("\nSaved Artifacts Directory:", str(video_out_dir))
    for k, v in result["artifacts"].items():
        print(f"  - {k}: {v}")
    print("=" * 75 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract cow muzzle groove lines and skeleton network from video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input cattle video file (e.g. muzzleVideo.mp4)")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--method", type=str, choices=["blackhat", "frangi"], default=config.GROOVE_ENHANCE_METHOD, help="Groove enhancement method")
    parser.add_argument("--scales", type=int, nargs="+", default=config.GROOVE_BLACKHAT_SCALES, help="Kernel scales (e.g. 9 15 21)")
    parser.add_argument("--threshold", type=str, choices=["otsu", "adaptive"], default=config.GROOVE_THRESHOLD_METHOD, help="Thresholding method")
    parser.add_argument("--min-size", type=int, default=config.GROOVE_MIN_COMPONENT_SIZE, help="Min connected component size in pixels")
    parser.add_argument("--exclude-nostrils", action="store_true", default=True, help="Exclude nostril cavities from groove detection")
    parser.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to Mask R-CNN weights file")

    args = parser.parse_args()
    process_video_grooves(
        video_path=args.video,
        output_dir=args.output_dir,
        method=args.method,
        scales=args.scales,
        threshold_method=args.threshold,
        min_component_size=args.min_size,
        exclude_nostrils=args.exclude_nostrils,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
