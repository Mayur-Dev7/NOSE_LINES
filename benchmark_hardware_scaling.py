#!/usr/bin/env python3
"""
Hardware Scaling Empirical Benchmark
====================================
Measures and records exact execution latencies for cattle muzzle detection & cropping
across arbitrary CPU core allocations:
- 2 Cores
- 4 Cores
- 5 Cores
- 6 Cores
- 12 Threads (Kernel Default)
- GPU (NVIDIA RTX 3050 Laptop)

Evaluates:
1. Single-Frame Still Image Latency Breakdown (Load, Preprocess, Inference, Postprocess, Total)
2. Video Frame Extraction Latency & End-to-End Throughput (FPS)
3. Scaling Efficiency & Speedup vs 2-Core Baseline
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = "/home/adis_mayur/F_CODES/crop11dec.h5"
PYTHON_EXEC = str(BASE_DIR / ".venv" / "bin" / "python")
CROPPER_SCRIPT = str(BASE_DIR / "muzzle_cropper.py")
MATCHER_SCRIPT = str(BASE_DIR / "muzzle_matcher.py")
OUTPUT_DIR = str(BASE_DIR / "output" / "benchmark_runs")

TEST_IMAGE = str(BASE_DIR / "test_samples" / "cow_muzzle_macro.jpg")
TEST_VIDEO = "/home/adis_mayur/F_CODES/muzzleVideo.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single_frame_test(device: str, cores: int = None, fast_mode: bool = True):
    """Runs single-frame detection and extracts timing breakdown."""
    out_dir = os.path.join(OUTPUT_DIR, f"single_{device}_{cores or 'all'}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON_EXEC,
        CROPPER_SCRIPT,
        "--weights", WEIGHTS_PATH,
        "--input", TEST_IMAGE,
        "--output-dir", out_dir,
        "--device", device,
        "--resize", "224",
        "--json-stream",
    ]
    if cores is not None:
        cmd.extend(["--cpu-cores", str(cores)])
    if fast_mode:
        cmd.append("--fast-mode")
    else:
        cmd.append("--no-fast-mode")

    env = os.environ.copy()
    if cores is not None:
        env["OMP_NUM_THREADS"] = str(cores)
        env["TF_NUM_INTRAOP_THREADS"] = str(cores)
        env["MKL_NUM_THREADS"] = str(cores)
        env["OPENBLAS_NUM_THREADS"] = str(cores)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    wall_duration = time.perf_counter() - t0

    timings = {}
    model_load_ms = 0.0
    for line in proc.stdout.splitlines():
        if "Model successfully loaded and initialized in" in line:
            try:
                model_load_ms = float(line.split("in")[1].replace("ms", "").strip())
            except Exception:
                pass
        if line.startswith("[JSON_EVENT]"):
            try:
                data = json.loads(line[len("[JSON_EVENT]"):].strip())
                if data.get("type") == "image":
                    timings = data.get("timings", {})
            except Exception:
                pass

    return {
        "status": "PASS" if proc.returncode == 0 and timings else "FAIL",
        "model_load_ms": round(model_load_ms, 1),
        "t_preprocess_ms": timings.get("t_preprocess_ms", 0.0),
        "t_inference_ms": timings.get("t_inference_ms", 0.0),
        "t_crop_ms": timings.get("t_crop_ms", 0.0),
        "t_total_ms": timings.get("t_total_ms", 0.0),
        "fps": timings.get("fps", 0.0),
        "wall_duration_s": round(wall_duration, 2),
        "error": proc.stderr[:200] if proc.returncode != 0 else None,
    }


def run_video_extraction_test(device: str, cores: int = None, max_frames: int = 3, fast_mode: bool = True):
    """Runs video frame processing across cores and extracts per-frame latencies."""
    out_dir = os.path.join(OUTPUT_DIR, f"video_{device}_{cores or 'all'}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON_EXEC,
        CROPPER_SCRIPT,
        "--weights", WEIGHTS_PATH,
        "--video", TEST_VIDEO,
        "--output-dir", out_dir,
        "--device", device,
        "--frame-interval", "5",
        "--max-frames", str(max_frames),
        "--resize", "224",
        "--json-stream",
    ]
    if cores is not None:
        cmd.extend(["--cpu-cores", str(cores)])
    if fast_mode:
        cmd.append("--fast-mode")
    else:
        cmd.append("--no-fast-mode")

    env = os.environ.copy()
    if cores is not None:
        env["OMP_NUM_THREADS"] = str(cores)
        env["TF_NUM_INTRAOP_THREADS"] = str(cores)
        env["MKL_NUM_THREADS"] = str(cores)
        env["OPENBLAS_NUM_THREADS"] = str(cores)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    wall_duration = time.perf_counter() - t0

    frame_inferences = []
    frame_totals = []
    summary_data = {}

    for line in proc.stdout.splitlines():
        if line.startswith("[JSON_EVENT]"):
            try:
                data = json.loads(line[len("[JSON_EVENT]"):].strip())
                if data.get("type") == "frame":
                    t = data.get("timings", {})
                    if "t_inference_ms" in t:
                        frame_inferences.append(t["t_inference_ms"])
                    if "t_total_ms" in t:
                        frame_totals.append(t["t_total_ms"])
                elif data.get("type") == "complete":
                    summary_data = data.get("summary", {})
            except Exception:
                pass

    avg_inf = sum(frame_inferences) / len(frame_inferences) if frame_inferences else 0.0
    avg_tot = sum(frame_totals) / len(frame_totals) if frame_totals else 0.0
    fps = 1000.0 / avg_tot if avg_tot > 0 else 0.0

    return {
        "status": "PASS" if proc.returncode == 0 and frame_inferences else "FAIL",
        "processed_frames": len(frame_inferences),
        "avg_inference_ms": round(avg_inf, 1),
        "avg_total_ms": round(avg_tot, 1),
        "avg_fps": round(fps, 2),
        "wall_duration_s": round(wall_duration, 2),
        "error": proc.stderr[:200] if proc.returncode != 0 else None,
    }


def main():
    print("=" * 85)
    print(" CATTLE MUZZLE DETECTION & HARDWARE SCHEDULING BENCHMARK")
    print(f" CPU: AMD Ryzen 5 5600H (6 Cores / 12 Threads) | GPU: NVIDIA RTX 3050 Laptop")
    print(f" Weights: {WEIGHTS_PATH}")
    print(f" Test Image: {TEST_IMAGE}")
    print(f" Test Video: {TEST_VIDEO}")
    print("=" * 85)

    profiles = [
        {"label": "CPU (2 Cores)", "device": "cpu", "cores": 2},
        {"label": "CPU (4 Cores)", "device": "cpu", "cores": 4},
        {"label": "CPU (5 Cores)", "device": "cpu", "cores": 5},
        {"label": "CPU (6 Cores)", "device": "cpu", "cores": 6},
        {"label": "CPU (Kernel Default - 12 Threads)", "device": "cpu", "cores": None},
        {"label": "GPU (RTX 3050 Laptop)", "device": "gpu", "cores": None},
    ]

    # PART 1: Still Frame Inference
    print("\n" + "#" * 85)
    print(" PART 1: STILL FRAME INFERENCE LATENCY BREAKDOWN (640x640 Fast Mode)")
    print("#" * 85)

    single_results = []
    base_inf = None

    for p in profiles:
        print(f"Testing {p['label']}...", flush=True)
        res = run_single_frame_test(device=p["device"], cores=p["cores"], fast_mode=True)
        res["label"] = p["label"]
        res["cores"] = p["cores"]
        res["device"] = p["device"]
        if p["cores"] == 2 and res["status"] == "PASS":
            base_inf = res["t_inference_ms"]
        single_results.append(res)
        print(f"  -> Model Load: {res['model_load_ms']}ms | Inference: {res['t_inference_ms']}ms | Total: {res['t_total_ms']}ms ({res['fps']} FPS)")

    # Compute speedup
    for r in single_results:
        if base_inf and r.get("t_inference_ms", 0) > 0:
            r["speedup_vs_2core"] = round(base_inf / r["t_inference_ms"], 2)
        else:
            r["speedup_vs_2core"] = 1.0

    # PART 2: Video Extraction Latency
    print("\n" + "#" * 85)
    print(" PART 2: VIDEO FRAME EXTRACTION & PIPELINE THROUGHPUT (3 frames sampled)")
    print("#" * 85)

    video_results = []
    base_vid_inf = None

    for p in profiles:
        print(f"Testing Video with {p['label']}...", flush=True)
        res = run_video_extraction_test(device=p["device"], cores=p["cores"], max_frames=3, fast_mode=True)
        res["label"] = p["label"]
        res["cores"] = p["cores"]
        res["device"] = p["device"]
        if p["cores"] == 2 and res["status"] == "PASS":
            base_vid_inf = res["avg_inference_ms"]
        video_results.append(res)
        print(f"  -> Frames: {res['processed_frames']} | Avg Inference: {res['avg_inference_ms']}ms | Avg Total: {res['avg_total_ms']}ms ({res['avg_fps']} FPS)")

    for r in video_results:
        if base_vid_inf and r.get("avg_inference_ms", 0) > 0:
            r["speedup_vs_2core"] = round(base_vid_inf / r["avg_inference_ms"], 2)
        else:
            r["speedup_vs_2core"] = 1.0

    # Summary Tables
    print("\n" + "=" * 95)
    print(f"{'SINGLE FRAME BENCHMARK':<35} | {'INFERENCE':<12} | {'TOTAL':<12} | {'FPS':<10} | {'SPEEDUP VS 2C':<14}")
    print("-" * 95)
    for r in single_results:
        print(f"{r['label']:<35} | {r['t_inference_ms']:>8.1f} ms | {r['t_total_ms']:>8.1f} ms | {r['fps']:>6.1f} FPS | {r['speedup_vs_2core']:>11.2f}x")
    print("=" * 95)

    print("\n" + "=" * 95)
    print(f"{'VIDEO EXTRACTION BENCHMARK':<35} | {'AVG INFERENCE':<14} | {'AVG TOTAL':<12} | {'AVG FPS':<10} | {'SPEEDUP VS 2C':<14}")
    print("-" * 95)
    for r in video_results:
        print(f"{r['label']:<35} | {r['avg_inference_ms']:>10.1f} ms | {r['avg_total_ms']:>8.1f} ms | {r['avg_fps']:>6.2f} FPS | {r['speedup_vs_2core']:>11.2f}x")
    print("=" * 95)

    # Save JSON summary for reference / UI
    bench_data = {
        "timestamp": time.time(),
        "hardware": {
            "cpu_model": "AMD Ryzen 5 5600H (6 physical cores / 12 logical threads)",
            "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB)",
        },
        "single_frame": single_results,
        "video_extraction": video_results,
    }
    json_path = os.path.join(OUTPUT_DIR, "hardware_scaling_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(bench_data, f, indent=2)
    print(f"\nSaved benchmark results to: {json_path}")


if __name__ == "__main__":
    main()
