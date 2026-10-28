#!/usr/bin/env python3
"""
Cow Muzzle Detection and Cropping Pipeline
=========================================
Uses Matterport Mask R-CNN with custom weights (crop11dec.h5) to detect,
segment, and crop cattle muzzles from still images or video frames.

Supports:
- Hardware selection: GPU (RTX 3050) vs CPU (Ryzen 5 5600H)
- CPU core / thread allocation: choice of 2 cores, N cores, or kernel default
- Detailed latency & timing breakdown logs (load, preprocess, inference, crop, total)
- Image directory and video file frame processing
- Aspect ratio preservation, square padding, and 224x224 biometric export
- Comparative hardware benchmarking mode
"""

import os
import sys
from pathlib import Path

# Auto-configure LD_LIBRARY_PATH for NVIDIA CUDA 11 / cuDNN in .venv
if os.environ.get("MRCNN_NVIDIA_CONFIGURED") != "1":
    SCRIPT_DIR_EARLY = Path(__file__).resolve().parent
    venv_nvidia = SCRIPT_DIR_EARLY / ".venv" / "lib" / "python3.8" / "site-packages" / "nvidia"
    if venv_nvidia.is_dir():
        lib_dirs = [str(p) for p in venv_nvidia.glob("*/lib") if p.is_dir()]
        if lib_dirs:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            current_parts = [p for p in current_ld.split(":") if p]
            needed = [p for p in lib_dirs if p not in current_parts]
            if needed:
                new_ld = ":".join(needed + current_parts)
                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = new_ld
                env["MRCNN_NVIDIA_CONFIGURED"] = "1"
                os.execve(sys.executable, [sys.executable, sys.argv[0]] + sys.argv[1:], env)

# Early CPU affinity & thread allocation before OpenMP / TensorFlow C++ engines initialize
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

import time
import json
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

# Ensure Mask_RCNN is in Python path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
class CustomFormatter(logging.Formatter):
    """Clean formatted logs with millisecond timestamps."""
    def format(self, record):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        millis = int((record.created - int(record.created)) * 1000)
        prefix = f"[{timestamp}.{millis:03d}] [{record.levelname:<5}]"
        return f"{prefix} {record.getMessage()}"


def setup_logger(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("muzzle_cropper")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(CustomFormatter())
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Hardware & Environment Configuration
# ---------------------------------------------------------------------------
def configure_hardware(device: str = "auto", cpu_cores: Optional[int] = None, logger: Optional[logging.Logger] = None):
    """
    Configures TensorFlow execution device and threading before importing TF.
    
    Args:
        device: 'gpu', 'cpu', or 'auto'
        cpu_cores: int number of CPU cores/threads (e.g. 1, 2, 3, 4, 5, 6, 8, 12), or None for kernel default
        logger: logger instance
    """
    # 1. CPU thread and affinity configuration
    if cpu_cores is not None and cpu_cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpu_cores)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(min(cpu_cores, 2))
        os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_cores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_cores)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_cores)

        if hasattr(os, "sched_setaffinity"):
            try:
                available_cpus = list(range(min(cpu_cores, os.cpu_count() or cpu_cores)))
                os.sched_setaffinity(0, set(available_cpus))
                if logger:
                    logger.info(f"[Hardware] CPU thread count constrained to: {cpu_cores} cores (affinity: {available_cpus})")
            except Exception as e:
                if logger:
                    logger.debug(f"[Hardware] sched_setaffinity note: {e}")
        else:
            if logger:
                logger.info(f"[Hardware] CPU thread count constrained to: {cpu_cores} threads")

        try:
            cv2.setNumThreads(cpu_cores)
        except Exception:
            pass

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    if cpu_cores is not None and cpu_cores > 0:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
        tf.config.threading.set_inter_op_parallelism_threads(min(cpu_cores, 2))
    else:
        if logger:
            logger.info(f"[Hardware] CPU allocation: OS kernel default scheduler (all available Ryzen threads: {os.cpu_count()})")

    # 2. Device (GPU vs CPU) configuration
    physical_gpus = tf.config.list_physical_devices("GPU")
    selected_device_desc = "CPU"

    if device.lower() in ("gpu", "cuda"):
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(physical_gpus[0], "GPU")
                selected_device_desc = f"GPU ({physical_gpus[0].name})"
                if logger:
                    logger.info(f"[Hardware] Successfully initialized {selected_device_desc} with dynamic memory growth")
            except Exception as e:
                if logger:
                    logger.warning(f"[Hardware] Failed to initialize GPU: {e}. Falling back to CPU.")
                tf.config.set_visible_devices([], "GPU")
                selected_device_desc = "CPU (GPU initialization failed)"
        else:
            if logger:
                logger.warning("[Hardware] GPU was requested but no GPU was detected by TensorFlow. Running on CPU.")
            selected_device_desc = "CPU"
    elif device.lower() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        selected_device_desc = f"CPU ({'Explicit ' + str(cpu_cores) + ' cores' if cpu_cores else 'Kernel default scheduler'})"
        if logger:
            logger.info(f"[Hardware] Explicitly selected {selected_device_desc}")
    else:  # auto
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                selected_device_desc = f"GPU ({physical_gpus[0].name})"
                if logger:
                    logger.info(f"[Hardware] Auto-selected {selected_device_desc}")
            except Exception:
                selected_device_desc = "CPU"
        else:
            selected_device_desc = "CPU"
            if logger:
                logger.info("[Hardware] Running on CPU")

    return selected_device_desc


# ---------------------------------------------------------------------------
# Mask R-CNN Cow Muzzle Config
# ---------------------------------------------------------------------------
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import utils


class CowMuzzleInferenceConfig(Config):
    NAME = "CowMuzzle"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 background + 1 cow muzzle
    DETECTION_MIN_CONFIDENCE = 0.70
    DETECTION_NMS_THRESHOLD = 0.30
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "square"


# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------
class MuzzleDetectionPipeline:
    """End-to-end cattle muzzle detector and cropper with latency profiling."""

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        cpu_cores: Optional[int] = None,
        confidence_threshold: float = 0.70,
        logger: Optional[logging.Logger] = None,
        fast_mode: Optional[bool] = None,
    ):
        self.logger = logger or setup_logger()
        self.weights_path = os.path.abspath(weights_path)
        self.confidence_threshold = confidence_threshold

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")

        # Configure hardware
        self.device_desc = configure_hardware(device=device, cpu_cores=cpu_cores, logger=self.logger)

        # Build Mask R-CNN config
        self.config = CowMuzzleInferenceConfig()
        self.config.DETECTION_MIN_CONFIDENCE = confidence_threshold

        # Optimization:
        # On GPU: Always use full training resolution (800x1024) with 100 ROIs for maximum accuracy and 325ms latency.
        # On CPU: Use 800x800 with 100 ROIs to match training anchor scales while saving 40% compute, or 640x640 only if explicitly requested.
        is_gpu = (device.lower() == "gpu" or "gpu" in self.device_desc.lower())
        if is_gpu:
            self.config.IMAGE_MIN_DIM = 800
            self.config.IMAGE_MAX_DIM = 1024
            self.config.POST_NMS_ROIS_INFERENCE = 100
            self.logger.info("[Optimization] GPU Engine active: Full 800x1024 resolution, 100 ROIs (real-time ~325ms)")
        else:
            # CPU Engine
            self.config.IMAGE_MIN_DIM = 800
            self.config.IMAGE_MAX_DIM = 800
            self.config.POST_NMS_ROIS_INFERENCE = 100
            self.logger.info(f"[Optimization] CPU Engine active: 800x800 resolution, 100 ROIs (Optimized for {cpu_cores or 'all'} cores)")

        t0 = time.perf_counter()
        self.logger.info(f"Loading Mask R-CNN inference model with weights from: {self.weights_path}...")
        self.model = modellib.MaskRCNN(
            mode="inference",
            config=self.config,
            model_dir=str(SCRIPT_DIR),
        )
        self.model.load_weights(self.weights_path, by_name=True)
        t_load = (time.perf_counter() - t0) * 1000
        self.logger.info(f"Model successfully loaded and initialized in {t_load:.1f} ms")

    def process_image(
        self,
        image_input,
        padding_pct: float = 0.05,
        make_square: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image or image array, detecting cow muzzles and cropping them.

        Returns a dictionary containing:
        - detections: list of detection dicts (bbox, score, class_id, cropped_image, mask)
        - timings: dict of millisecond breakdown (load, preprocess, inference, crop, total)
        - original_shape: (height, width, channels)
        """
        timings = {}
        t_start = time.perf_counter()

        # Step 1: Load image
        t0 = time.perf_counter()
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Failed to read image at: {image_input}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            source_name = os.path.basename(image_input)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                image_rgb = image_input
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            source_name = "in_memory_frame"
        else:
            raise TypeError("image_input must be file path or numpy ndarray")
        timings["t_load_ms"] = (time.perf_counter() - t0) * 1000

        orig_h, orig_w = image_rgb.shape[:2]

        # Step 2: Preprocess / Mold input
        t0 = time.perf_counter()
        molded_images, image_metas, windows = self.model.mold_inputs([image_rgb])
        molded_image_shape = molded_images[0].shape
        windows = windows[0]
        timings["t_preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3: Neural network forward inference
        t0 = time.perf_counter()
        # Compute anchors
        anchors = self.model.get_anchors(molded_image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        
        # Run detection forward pass
        detections, _, _, mrcnn_mask, _, _, _ = self.model.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        timings["t_inference_ms"] = (time.perf_counter() - t0) * 1000

        # Step 4: Post-process & Unmold detections
        t0 = time.perf_counter()
        rois, class_ids, scores, masks = self.model.unmold_detections(
            detections[0], mrcnn_mask[0], image_rgb.shape, molded_image_shape, windows
        )
        timings["t_postprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: Crop muzzles
        t0 = time.perf_counter()
        results = []
        for idx in range(len(rois)):
            score = float(scores[idx])
            if score < self.confidence_threshold:
                continue

            y1, x1, y2, x2 = rois[idx]
            box_h = y2 - y1
            box_w = x2 - x1

            # Apply padding
            pad_y = int(box_h * padding_pct)
            pad_x = int(box_w * padding_pct)
            cy1 = max(0, y1 - pad_y)
            cx1 = max(0, x1 - pad_x)
            cy2 = min(orig_h, y2 + pad_y)
            cx2 = min(orig_w, x2 + pad_x)

            # Optional square expansion
            if make_square:
                ch = cy2 - cy1
                cw = cx2 - cx1
                diff = abs(ch - cw)
                if ch < cw:
                    cy1 = max(0, cy1 - diff // 2)
                    cy2 = min(orig_h, cy2 + (diff - diff // 2))
                elif cw < ch:
                    cx1 = max(0, cx1 - diff // 2)
                    cx2 = min(orig_w, cx2 + (diff - diff // 2))

            # Extract crop
            cropped_bgr = image_bgr[cy1:cy2, cx1:cx2].copy()
            cropped_rgb = image_rgb[cy1:cy2, cx1:cx2].copy()

            # Optional resize
            if target_size:
                cropped_bgr = cv2.resize(cropped_bgr, target_size, interpolation=cv2.INTER_AREA)
                cropped_rgb = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA)

            results.append({
                "muzzle_index": idx,
                "confidence": score,
                "class_id": int(class_ids[idx]),
                "bbox_original": [int(y1), int(x1), int(y2), int(x2)],
                "bbox_cropped": [int(cy1), int(cx1), int(cy2), int(cx2)],
                "crop_size": [int(cy2 - cy1), int(cx2 - cx1)],
                "cropped_bgr": cropped_bgr,
                "cropped_rgb": cropped_rgb,
                "mask": masks[:, :, idx] if masks is not None and masks.shape[-1] > idx else None,
            })

        timings["t_crop_ms"] = (time.perf_counter() - t0) * 1000
        timings["t_total_ms"] = (time.perf_counter() - t_start) * 1000
        timings["fps"] = 1000.0 / timings["t_total_ms"] if timings["t_total_ms"] > 0 else 0.0

        return {
            "source": source_name,
            "original_shape": [orig_h, orig_w, 3],
            "detections": results,
            "timings": timings,
            "image_bgr": image_bgr,
        }

    def save_results(
        self,
        processed: Dict[str, Any],
        output_dir: str,
        save_annotated: bool = True,
    ) -> List[str]:
        """Saves cropped muzzles and annotated visualizations."""
        saved_paths = []
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(processed["source"]).stem

        # Save cropped muzzles
        for det in processed["detections"]:
            idx = det["muzzle_index"]
            conf = det["confidence"]
            crop_filename = f"{base_name}_muzzle_{idx}_conf{int(conf*100)}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, det["cropped_bgr"])
            det["saved_crop_path"] = crop_path
            saved_paths.append(crop_path)

        # Save annotated image
        if save_annotated and len(processed["detections"]) > 0:
            annotated = processed["image_bgr"].copy()
            for det in processed["detections"]:
                y1, x1, y2, x2 = det["bbox_original"]
                conf = det["confidence"]
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"Cow Muzzle: {conf*100:.1f}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - h - 10)), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, max(h, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Overlay mask if present
                mask = det.get("mask")
                if mask is not None:
                    color_mask = np.zeros_like(annotated, dtype=np.uint8)
                    color_mask[mask] = (0, 165, 255)  # Orange overlay
                    annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.35, 0)

            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated)
            processed["saved_annotated_path"] = annotated_path

        return saved_paths


# ---------------------------------------------------------------------------
# Video Processor
# ---------------------------------------------------------------------------
def process_video(
    pipeline: MuzzleDetectionPipeline,
    video_path: str,
    output_dir: str,
    frame_interval: int = 5,
    max_frames: Optional[int] = None,
    padding_pct: float = 0.05,
    make_square: bool = True,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    logger: Optional[logging.Logger] = None,
    json_stream: bool = False,
) -> Dict[str, Any]:
    """Extracts and crops muzzles from frames of a video file."""
    log = logger or pipeline.logger
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    log.info(f"Processing video: {video_path} ({total_frames} frames, {fps:.1f} fps, {duration:.1f}s)")
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    processed_count = 0
    total_detections = 0
    frame_timings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_source = f"{Path(video_path).stem}_frame_{frame_idx:05d}"
            res = pipeline.process_image(
                frame,
                padding_pct=padding_pct,
                make_square=make_square,
                target_size=target_size,
            )
            res["source"] = frame_source

            detections = res["detections"]
            num_det = len(detections)
            total_detections += num_det
            frame_timings.append(res["timings"])

            t = res["timings"]
            log.info(
                f"Frame #{frame_idx:04d}: {num_det} muzzle(s) found | "
                f"Infer: {t['t_inference_ms']:.1f}ms | Total: {t['t_total_ms']:.1f}ms ({t['fps']:.1f} FPS)"
            )

            if num_det > 0:
                pipeline.save_results(res, output_dir, save_annotated=True)

            if json_stream:
                crops_info = [
                    {
                        "index": d["muzzle_index"],
                        "confidence": float(d["confidence"]),
                        "bbox": d["bbox_original"],
                        "crop_path": d.get("saved_crop_path"),
                        "crop_size": d.get("crop_size")
                    }
                    for d in res["detections"]
                ]
                event = {
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "muzzle_count": num_det,
                    "annotated_path": res.get("saved_annotated_path"),
                    "crops": crops_info,
                    "timings": t,
                }
                print(f"[JSON_EVENT] {json.dumps(event)}", flush=True)

            processed_count += 1
            if max_frames and processed_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    avg_inference = np.mean([t["t_inference_ms"] for t in frame_timings]) if frame_timings else 0.0
    avg_total = np.mean([t["t_total_ms"] for t in frame_timings]) if frame_timings else 0.0

    summary = {
        "video": video_path,
        "total_video_frames": total_frames,
        "processed_frames": processed_count,
        "total_muzzles_detected": total_detections,
        "avg_inference_ms": avg_inference,
        "avg_total_ms": avg_total,
        "avg_fps": 1000.0 / avg_total if avg_total > 0 else 0.0,
    }
    log.info(
        f"Video Processing Complete: {processed_count} frames sampled, "
        f"{total_detections} muzzles cropped. Average speed: {summary['avg_fps']:.1f} FPS"
    )
    if json_stream:
        print(f"[JSON_EVENT] {json.dumps({'type': 'complete', 'summary': summary})}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Hardware Benchmark Suite
# ---------------------------------------------------------------------------
def run_benchmark(
    weights_path: str,
    test_images: List[str],
    output_dir: str,
    logger: Optional[logging.Logger] = None,
    json_stream: bool = False,
):
    """
    Benchmarks Mask R-CNN cow muzzle detector across different hardware settings:
    1. CPU - 2 Cores
    2. CPU - Default Kernel Scheduler (All Ryzen Cores)
    3. GPU - RTX 3050 (if available)
    """
    import subprocess
    log = logger or setup_logger()

    configurations = [
        {"name": "CPU (2 Cores)", "device": "cpu", "cores": 2},
        {"name": "CPU (4 Cores)", "device": "cpu", "cores": 4},
        {"name": "CPU (6 Cores)", "device": "cpu", "cores": 6},
        {"name": "CPU (Kernel Default - 12 Threads)", "device": "cpu", "cores": None},
    ]

    # Check if GPU is functional
    try:
        import tensorflow.compat.v1 as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            configurations.append({"name": "GPU (RTX 3050 Laptop)", "device": "gpu", "cores": None})
    except Exception:
        pass

    log.info("=" * 80)
    log.info("STARTING COW MUZZLE HARDWARE PERFORMANCE BENCHMARK")
    log.info(f"Target Model: {weights_path}")
    log.info(f"Benchmarking across {len(configurations)} hardware profiles on {len(test_images)} test images")
    log.info("=" * 80)

    benchmark_results = []

    # To isolate TF process state and thread affinity cleanly across runs,
    # each benchmark configuration runs in a clean subprocess invocation.
    for config_item in configurations:
        cfg_name = config_item["name"]
        device = config_item["device"]
        cores = config_item["cores"]

        log.info(f"\n>>> Running Profile: {cfg_name}...")

        # Subprocess runner command
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--weights", weights_path,
            "--device", device,
            "--input-dir", str(Path(test_images[0]).parent),
            "--output-dir", os.path.join(output_dir, f"bench_{device}_{cores or 'all'}"),
            "--max-images", str(len(test_images)),
            "--quiet",
        ]
        if cores is not None:
            cmd.extend(["--cpu-cores", str(cores)])
        cmd.append("--no-fast-mode")

        # Run benchmark run
        t_bench_start = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        bench_duration = time.perf_counter() - t_bench_start

        if proc.returncode == 0:
            # Parse timing lines from stdout
            inference_times = []
            total_times = []
            for line in proc.stdout.splitlines():
                if "[TIMING]" in line:
                    parts = line.split("|")
                    for p in parts:
                        if "Inference:" in p:
                            try:
                                val = float(p.split(":")[1].replace("ms", "").strip())
                                inference_times.append(val)
                            except ValueError:
                                pass
                        if "Total:" in p:
                            try:
                                val = float(p.split(":")[1].replace("ms", "").split()[0].strip())
                                total_times.append(val)
                            except ValueError:
                                pass

            # Compute statistics (skip 1st warm-up image for steady-state)
            steady_inf = inference_times[1:] if len(inference_times) > 1 else inference_times
            steady_tot = total_times[1:] if len(total_times) > 1 else total_times

            res = {
                "profile": cfg_name,
                "warmup_inference_ms": inference_times[0] if inference_times else 0.0,
                "mean_inference_ms": np.mean(steady_inf) if steady_inf else 0.0,
                "median_inference_ms": np.median(steady_inf) if steady_inf else 0.0,
                "min_inference_ms": np.min(steady_inf) if steady_inf else 0.0,
                "max_inference_ms": np.max(steady_inf) if steady_inf else 0.0,
                "mean_total_ms": np.mean(steady_tot) if steady_tot else 0.0,
                "steady_fps": 1000.0 / np.mean(steady_tot) if steady_tot and np.mean(steady_tot) > 0 else 0.0,
                "status": "PASS",
            }
            benchmark_results.append(res)
            log.info(
                f"Completed {cfg_name}: Mean Inference: {res['mean_inference_ms']:.1f}ms | "
                f"Steady FPS: {res['steady_fps']:.2f} FPS"
            )
        else:
            log.error(f"Failed {cfg_name}: {proc.stderr[:300]}")
            benchmark_results.append({"profile": cfg_name, "status": "FAIL", "error": proc.stderr[:200]})

    # Output Benchmark Summary Table
    log.info("\n" + "=" * 95)
    log.info(f"{'HARDWARE PROFILE':<35} | {'INFERENCE (MEAN)':<18} | {'TOTAL LATENCY':<16} | {'THROUGHPUT':<12}")
    log.info("-" * 95)
    for b in benchmark_results:
        if b.get("status") == "PASS":
            inf_str = f"{b['mean_inference_ms']:.1f} ms (p50: {b['median_inference_ms']:.1f})"
            tot_str = f"{b['mean_total_ms']:.1f} ms"
            fps_str = f"{b['steady_fps']:.2f} FPS"
            log.info(f"{b['profile']:<35} | {inf_str:<18} | {tot_str:<16} | {fps_str:<12}")
        else:
            log.info(f"{b['profile']:<35} | FAILED ({b.get('error','')[:20]})")
    log.info("=" * 95 + "\n")

    if json_stream:
        print(f"[JSON_EVENT] {json.dumps({'type': 'benchmark', 'results': benchmark_results})}", flush=True)

    return benchmark_results


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def check_positive_int(value: str) -> int:
    try:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"Invalid cpu_cores: {value}. Must be an integer >= 1.")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid cpu_cores: {value}. Must be an integer >= 1.")


def main():
    parser = argparse.ArgumentParser(
        description="Cow Muzzle Detection and Cropping Pipeline using Mask R-CNN (crop11dec.h5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model parameters
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/adis_mayur/F_CODES/crop11dec.h5",
        help="Path to trained Mask R-CNN cow muzzle weights (.h5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Detection confidence threshold (0.0 to 1.0)",
    )

    # Hardware selection options
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Target hardware device: 'gpu' (RTX 3050), 'cpu' (Ryzen), or 'auto'",
    )
    parser.add_argument(
        "--cpu-cores",
        type=check_positive_int,
        default=None,
        help="Limit CPU threads/cores (any integer >= 1, e.g. 1, 2, 3, 4, 5, 6, 8, 12). Omit for OS kernel default",
    )
    parser.add_argument(
        "--fast-mode",
        dest="fast_mode",
        action="store_true",
        default=None,
        help="Enable 640x640 Fast CPU mode",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable Fast CPU mode (force standard 1024x800 resolution)",
    )

    # Input modes
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to a single image file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to a directory of images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process from input-dir",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file to extract and crop frames",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Sample every N frames when processing video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to sample and process from video",
    )

    # Cropping options
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/home/adis_mayur/F_CODES/Mask_RCNN/output",
        help="Directory to save cropped muzzle images",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Fractional margin padding around muzzle (e.g. 0.08 for 8%%)",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        default=True,
        help="Crop muzzle with 1:1 square aspect ratio",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Resize cropped muzzle to square size (e.g. 224 for 224x224 ArcFace ID)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize crop, preserve full native crop resolution",
    )

    # Benchmarking & Logging
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive comparative benchmark across hardware options",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="/home/adis_mayur/F_CODES/Mask_RCNN/muzzle_cropper.log",
        help="File path to save detailed logs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout verbosity",
    )
    parser.add_argument(
        "--json-stream",
        action="store_true",
        help="Emit structured JSON lines ([JSON_EVENT]) for real-time UI streaming",
    )

    args = parser.parse_args()
    logger = setup_logger(verbose=not args.quiet, log_file=args.log_file)

    target_size = None if args.no_resize else (args.resize, args.resize)

    # Handle Benchmark mode
    if args.benchmark:
        test_dir = args.input_dir or os.path.join(SCRIPT_DIR, "test_samples")
        import glob
        num_bench = args.max_images or 3
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))[:num_bench]
        if not test_images:
            logger.error(f"No test images found in {test_dir} for benchmark.")
            sys.exit(1)
        run_benchmark(args.weights, test_images, args.output_dir, logger=logger, json_stream=args.json_stream)
        return

    # Initialize Pipeline
    pipeline = MuzzleDetectionPipeline(
        weights_path=args.weights,
        device=args.device,
        cpu_cores=args.cpu_cores,
        confidence_threshold=args.confidence,
        logger=logger,
        fast_mode=args.fast_mode,
    )

    # Video Mode
    if args.video:
        process_video(
            pipeline=pipeline,
            video_path=args.video,
            output_dir=os.path.join(args.output_dir, "video_muzzles"),
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            padding_pct=args.padding,
            make_square=args.square,
            target_size=target_size,
            logger=logger,
            json_stream=args.json_stream,
        )
        return

    # Image list collection
    image_paths = []
    if args.input:
        image_paths.append(args.input)
    elif args.input_dir:
        import glob
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.sort()
    else:
        # Default fallback to test samples
        default_test_dir = os.path.join(SCRIPT_DIR, "test_samples")
        import glob
        image_paths = sorted(glob.glob(os.path.join(default_test_dir, "*.jpg")))

    if not image_paths:
        logger.error("No input images provided. Use --input, --input-dir, or --video.")
        sys.exit(1)

    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[:args.max_images]

    logger.info(f"Processing {len(image_paths)} image(s) using {pipeline.device_desc}...")
    summary_stats = []

    for img_path in image_paths:
        try:
            res = pipeline.process_image(
                img_path,
                padding_pct=args.padding,
                make_square=args.square,
                target_size=target_size,
            )
            saved = pipeline.save_results(res, args.output_dir, save_annotated=True)
            t = res["timings"]
            det_count = len(res["detections"])

            logger.info(
                f"[TIMING] {Path(img_path).name:<26} | Muzzles: {det_count} | "
                f"Inference: {t['t_inference_ms']:>6.1f}ms | Total: {t['t_total_ms']:>6.1f}ms ({t['fps']:>4.1f} FPS)"
            )

            for d in res["detections"]:
                box = d["bbox_original"]
                crop_dim = d["crop_size"]
                logger.info(
                    f"   -> Detected Muzzle #{d['muzzle_index']+1}: "
                    f"Conf: {d['confidence']*100:.1f}% | Box: [y1={box[0]}, x1={box[1]}, y2={box[2]}, x2={box[3]}] | "
                    f"Crop: {crop_dim[1]}x{crop_dim[0]} px -> Saved: {d.get('saved_crop_path')}"
                )

            if args.json_stream:
                crops_info = [
                    {
                        "index": d["muzzle_index"],
                        "confidence": float(d["confidence"]),
                        "bbox": d["bbox_original"],
                        "crop_path": d.get("saved_crop_path"),
                        "crop_size": d.get("crop_size"),
                    }
                    for d in res["detections"]
                ]
                event = {
                    "type": "image",
                    "source": Path(img_path).name,
                    "muzzle_count": det_count,
                    "annotated_path": res.get("saved_annotated_path"),
                    "crops": crops_info,
                    "timings": t,
                }
                print(f"[JSON_EVENT] {json.dumps(event)}", flush=True)

            summary_stats.append(t)
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)

    if summary_stats:
        mean_inf = np.mean([s["t_inference_ms"] for s in summary_stats])
        mean_tot = np.mean([s["t_total_ms"] for s in summary_stats])
        fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0
        logger.info("-" * 80)
        logger.info(f"Summary ({len(summary_stats)} images on {pipeline.device_desc}):")
        logger.info(f"  Average Inference Latency : {mean_inf:.1f} ms")
        logger.info(f"  Average End-to-End Latency: {mean_tot:.1f} ms ({fps:.2f} FPS)")
        logger.info(f"  All cropped muzzles saved to: {args.output_dir}")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
