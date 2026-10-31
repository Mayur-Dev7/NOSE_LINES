#!/usr/bin/env python3
"""
benchmark_pipeline.py
=====================
Official benchmark harness for Bovine Biometric verification pipeline.
Profiles: GPU, CPU-6, CPU-2
Evaluates muzzleVideo.mp4 vs muzzleVideo2.mp4
Protocol: 1 warm-up run, followed by 3 timed runs.
Measures: median and mean per-stage milliseconds (Steps 1-7) and total wall-clock seconds.
Can also save or verify golden artifacts.
"""

import os
import sys
import time
import json
import fcntl
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_groove_extractor import (
    extract_five_muzzle_stages,
    extract_two_videos_biometric_stages,
    compute_cosine_similarity,
)
from server import resolve_video_path

LOCK_PATH = "/tmp/nose_bench.lock"


def run_single_comparison(
    video_a_path: str,
    video_b_path: str,
    device: str,
    cpu_cores: int,
    save_artifacts_dir: Path = None,
) -> Dict[str, Any]:
    """Runs a complete end-to-end two-video comparison and returns measured step timings and artifacts."""
    resolved_a = resolve_video_path(video_a_path)
    resolved_b = resolve_video_path(video_b_path)
    if not resolved_a or not resolved_a.exists():
        raise FileNotFoundError(f"Video A not found: {video_a_path}")
    if not resolved_b or not resolved_b.exists():
        raise FileNotFoundError(f"Video B not found: {video_b_path}")

    t_wall_start = time.perf_counter()

    # Step 1-6 for Video A and Video B (pipelined on GPU, sequential on CPU)
    res_a, res_b = extract_two_videos_biometric_stages(
        video_a_path=str(resolved_a),
        video_b_path=str(resolved_b),
        device=device,
        cpu_cores=cpu_cores,
    )
    t_a_wall = res_a.get("step_timings", {}).get("t_total_ms", 0.0)
    t_b_wall = res_b.get("step_timings", {}).get("t_total_ms", 0.0)

    # Step 7: Biometric Consensus & Verification
    t_start_7 = time.perf_counter()
    vec_a = res_a.get("vector_array") or list(res_a["median_parameters"].values())
    vec_b = res_b.get("vector_array") or list(res_b["median_parameters"].values())
    dl_emb_a = res_a.get("dl_embedding", {}).get("median_embedding") if res_a.get("dl_embedding") else None
    dl_emb_b = res_b.get("dl_embedding", {}).get("median_embedding") if res_b.get("dl_embedding") else None
    per_frames_a = res_a.get("dl_embedding", {}).get("per_frame_embeddings") if res_a.get("dl_embedding") else None
    per_frames_b = res_b.get("dl_embedding", {}).get("per_frame_embeddings") if res_b.get("dl_embedding") else None

    sim_res = compute_cosine_similarity(
        vec_a=vec_a,
        vec_b=vec_b,
        dl_emb_a=dl_emb_a,
        dl_emb_b=dl_emb_b,
        per_frames_a=per_frames_a,
        per_frames_b=per_frames_b,
    )
    t_7_wall = (time.perf_counter() - t_start_7) * 1000.0

    t_wall_total = (time.perf_counter() - t_wall_start)

    timings_a = res_a.get("step_timings", {})
    timings_b = res_b.get("step_timings", {})
    t_7_ms = sim_res.get("t_consensus_ms", t_7_wall)

    stage_timings = {
        "step_1_sampling_ms": round(float(timings_a.get("t_sampling_ms", 0.0) + timings_b.get("t_sampling_ms", 0.0)), 2),
        "step_2_detector_ms": round(float(timings_a.get("t_detector_ms", 0.0) + timings_b.get("t_detector_ms", 0.0)), 2),
        "step_3_alignment_ms": round(float(timings_a.get("t_canonical_clahe_ms", 0.0) + timings_b.get("t_canonical_clahe_ms", 0.0)), 2),
        "step_4_groove_ms": round(float(timings_a.get("t_groove_extract_ms", 0.0) + timings_b.get("t_groove_extract_ms", 0.0)), 2),
        "step_5_skeleton_ms": round(float(timings_a.get("t_skeleton_minutiae_ms", 0.0) + timings_b.get("t_skeleton_minutiae_ms", 0.0)), 2),
        "step_6_embedding_ms": round(float(timings_a.get("t_dl_inference_ms", 0.0) + timings_b.get("t_dl_inference_ms", 0.0)), 2),
        "step_7_consensus_ms": round(float(t_7_ms), 2),
        "total_wall_clock_s": round(float(t_wall_total), 2),
        "video_a_wall_ms": round(float(t_a_wall), 2),
        "video_b_wall_ms": round(float(t_b_wall), 2),
    }

    # Save golden artifacts if requested
    if save_artifacts_dir:
        save_artifacts_dir.mkdir(parents=True, exist_ok=True)
        # 1. Embeddings and Tensors (per-frame 3x224x224 and 512-D)
        if per_frames_a:
            np.save(save_artifacts_dir / "video_a_embeddings.npy", np.array(per_frames_a, dtype=np.float32))
        if per_frames_b:
            np.save(save_artifacts_dir / "video_b_embeddings.npy", np.array(per_frames_b, dtype=np.float32))
        tensors_a = res_a.get("dl_tensors", [])
        tensors_b = res_b.get("dl_tensors", [])
        if tensors_a:
            np.save(save_artifacts_dir / "video_a_tensors.npy", np.array(tensors_a, dtype=np.float32))
        if tensors_b:
            np.save(save_artifacts_dir / "video_b_tensors.npy", np.array(tensors_b, dtype=np.float32))

        # 2. Graph minutiae features (P_graph)
        with open(save_artifacts_dir / "video_a_p_graph.json", "w") as f:
            json.dump(res_a.get("median_parameters", {}), f, indent=2)
        with open(save_artifacts_dir / "video_b_p_graph.json", "w") as f:
            json.dump(res_b.get("median_parameters", {}), f, indent=2)

        # 3. Detector boxes and masks
        frames_a = res_a.get("frames", [])
        frames_b = res_b.get("frames", [])
        boxes_a = [f.get("bbox", []) for f in frames_a]
        boxes_b = [f.get("bbox", []) for f in frames_b]
        masks_a = [f.get("mask") for f in frames_a if f.get("mask") is not None]
        masks_b = [f.get("mask") for f in frames_b if f.get("mask") is not None]
        np.savez_compressed(
            save_artifacts_dir / "detector_artifacts.npz",
            boxes_a=np.array(boxes_a, dtype=object),
            boxes_b=np.array(boxes_b, dtype=object),
            masks_a=np.array(masks_a, dtype=object),
            masks_b=np.array(masks_b, dtype=object),
        )

        # 4. Final verification results (S_calibrated, raw cosine, verdict)
        verif_data = {
            "cosine_similarity": sim_res.get("cosine_similarity"),
            "raw_cosine_similarity": sim_res.get("raw_cosine_similarity"),
            "magnitude_penalty_factor": sim_res.get("magnitude_penalty_factor"),
            "dl_similarity_pct": sim_res.get("dl_similarity_pct"),
            "mean_pooling_similarity_pct": sim_res.get("mean_pooling_similarity_pct"),
            "median_pooling_similarity_pct": sim_res.get("median_pooling_similarity_pct"),
            "is_match": sim_res.get("is_match"),
            "verdict": sim_res.get("verdict"),
        }
        with open(save_artifacts_dir / "verification_result.json", "w") as f:
            json.dump(verif_data, f, indent=2)

    return {
        "stage_timings": stage_timings,
        "verification": {
            "cosine_similarity": sim_res.get("cosine_similarity"),
            "dl_similarity_pct": sim_res.get("dl_similarity_pct"),
            "verdict": sim_res.get("verdict"),
        },
        "res_a": res_a,
        "res_b": res_b,
        "sim_res": sim_res,
    }


def benchmark_profile(
    profile_name: str,
    device: str,
    cpu_cores: int,
    video_a: str,
    video_b: str,
    save_golden: bool = False,
    golden_dir: Path = None,
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"BENCHMARKING PROFILE: {profile_name} (device={device}, cores={cpu_cores})")
    print(f"==================================================")

    # 1. Warm-up run
    print("[1/4] Running warm-up run (not included in official stats)...")
    t0 = time.time()
    warmup_res = run_single_comparison(video_a, video_b, device, cpu_cores)
    print(f"      Warm-up completed in {time.time() - t0:.2f}s (wall-clock: {warmup_res['stage_timings']['total_wall_clock_s']}s)")

    # 2. Three timed benchmark runs
    runs = []
    for i in range(3):
        print(f"[{i+2}/4] Timed run {i+1} of 3...")
        # If saving golden, save from run 1
        art_dir = (golden_dir / profile_name) if (save_golden and i == 0 and golden_dir) else None
        res = run_single_comparison(video_a, video_b, device, cpu_cores, save_artifacts_dir=art_dir)
        runs.append(res["stage_timings"])
        print(f"      Run {i+1} completed in {res['stage_timings']['total_wall_clock_s']}s")

    # Aggregate stats: median and mean
    stage_keys = [
        "step_1_sampling_ms",
        "step_2_detector_ms",
        "step_3_alignment_ms",
        "step_4_groove_ms",
        "step_5_skeleton_ms",
        "step_6_embedding_ms",
        "step_7_consensus_ms",
        "total_wall_clock_s",
    ]

    median_stats = {}
    mean_stats = {}
    for k in stage_keys:
        vals = [r[k] for r in runs]
        median_stats[k] = round(float(np.median(vals)), 2)
        mean_stats[k] = round(float(np.mean(vals)), 2)

    total_time_ms = sum(median_stats[k] for k in stage_keys if k != "total_wall_clock_s")
    share_pct = {}
    for k in stage_keys:
        if k != "total_wall_clock_s":
            share_pct[k] = round((median_stats[k] / max(1e-4, total_time_ms)) * 100.0, 1)

    print("\n--- RESULTS FOR " + profile_name + " ---")
    print(f"{'Stage':<25} | {'Median (ms)':<12} | {'Mean (ms)':<12} | {'Share (%)':<10}")
    print("-" * 65)
    for k in stage_keys:
        if k != "total_wall_clock_s":
            print(f"{k:<25} | {median_stats[k]:<12.1f} | {mean_stats[k]:<12.1f} | {share_pct.get(k, 0.0):<10.1f}%")
    print("-" * 65)
    print(f"{'Total Wall-Clock (s)':<25} | {median_stats['total_wall_clock_s']:<12.2f} | {mean_stats['total_wall_clock_s']:<12.2f} | 100.0%")

    return {
        "profile": profile_name,
        "device": device,
        "cpu_cores": cpu_cores,
        "runs": runs,
        "median": median_stats,
        "mean": mean_stats,
        "share_pct": share_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Bovine Biometric Pipeline Benchmark")
    parser.add_argument("--profiles", nargs="+", default=["GPU", "CPU-6", "CPU-2"], help="Profiles to benchmark")
    parser.add_argument("--save-golden", action="store_true", help="Save golden artifacts for regression testing")
    parser.add_argument("--golden-dir", default=str(PROJECT_ROOT / "golden_artifacts"), help="Directory to store golden artifacts")
    parser.add_argument("--video-a", default="muzzleVideo.mp4", help="Video A filename or path")
    parser.add_argument("--video-b", default="muzzleVideo2.mp4", help="Video B filename or path")
    parser.add_argument("--output-json", default=None, help="Save summary benchmark results to JSON")
    args = parser.parse_args()

    # Coordination lock
    lock_file = open(LOCK_PATH, "w")
    need_unlock = False
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need_unlock = True
        print(f"Acquired exclusive benchmark lock: {LOCK_PATH}")
    except (BlockingIOError, IOError):
        print(f"Parent shell already holds exclusive benchmark lock: {LOCK_PATH}")

    try:
        profile_configs = {
            "GPU": {"device": "gpu", "cpu_cores": 6},
            "CPU-6": {"device": "cpu", "cpu_cores": 6},
            "CPU-2": {"device": "cpu", "cpu_cores": 2},
        }

        all_results = {}
        for prof in args.profiles:
            if prof not in profile_configs:
                print(f"Unknown profile: {prof}")
                continue
            cfg = profile_configs[prof]
            res = benchmark_profile(
                profile_name=prof,
                device=cfg["device"],
                cpu_cores=cfg["cpu_cores"],
                video_a=args.video_a,
                video_b=args.video_b,
                save_golden=args.save_golden,
                golden_dir=Path(args.golden_dir),
            )
            all_results[prof] = res

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nAll benchmark results successfully written to: {out_path}")

    finally:
        if need_unlock:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                print("Coordination lock released.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
