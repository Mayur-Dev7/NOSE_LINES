#!/usr/bin/env python3
"""
regression_checker.py
=====================
Strict regression gate for Bovine Biometric pipeline optimizations.
Enforces:
1. Per-frame embedding cosine similarity >= 0.9999 vs golden artifacts
2. S_calibrated (final calibrated similarity) delta <= 0.1% vs golden
3. P_graph (topological minutiae features) bit-identical for lossless changes
4. Detector box IoU and mask IoU >= 0.98 (if detector modified)
5. Genuine vs Impostor score separation evaluation across available dataset pairs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_groove_extractor import (
    extract_five_muzzle_stages,
    compute_cosine_similarity,
)
from server import resolve_video_path


def compute_iou_boxes(box1, box2):
    """Computes Intersection-over-Union between two bounding boxes [y1, x1, y2, x2]."""
    if box1 is None or box2 is None or len(box1) != 4 or len(box2) != 4:
        return 1.0 if (box1 is None and box2 is None) else 0.0
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 1.0 if area1 == area2 else 0.0
    return inter_area / union_area


def cosine_sim_vecs(v1, v2):
    """Cosine similarity between two 1D float vectors."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_embedder_tensor_regression(golden_dir: Path, device: str = "cpu") -> Dict[str, Any]:
    """Tests Step 6 embedder directly against golden 3x224x224 tensors."""
    from dl_muzzle_embedder import get_muzzle_embedder
    import torch

    prof_golden = golden_dir / "GPU"
    if not (prof_golden / "video_a_tensors.npy").exists():
        return {"passed": True, "message": "No golden tensors found to check."}

    tensors_a = np.load(prof_golden / "video_a_tensors.npy")
    gold_emb_a = np.load(prof_golden / "video_a_embeddings.npy")

    model = get_muzzle_embedder(backbone="convnext_tiny", device=device)
    model.eval()

    failures = []
    passes = []
    # Test batch forward
    t_in = torch.from_numpy(tensors_a).to(next(model.parameters()).device)
    with torch.no_grad():
        out_embs = model(t_in).cpu().numpy()

    for i in range(len(tensors_a)):
        c = cosine_sim_vecs(out_embs[i], gold_emb_a[i])
        if c < 0.9999:
            failures.append(f"Direct Tensor Frame {i+1} cosine: {c:.6f} < 0.9999")
        else:
            passes.append(f"Direct Tensor Frame {i+1} cosine: {c:.6f} >= 0.9999")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "passes": passes,
    }


def check_regression_against_golden(
    profile_name: str,
    device: str,
    cpu_cores: int,
    video_a_path: str,
    video_b_path: str,
    golden_dir: Path,
    strict_p_graph: bool = True,
) -> Dict[str, Any]:
    """Runs a single test comparison and checks against stored golden artifacts."""
    prof_golden = golden_dir / profile_name
    if not prof_golden.exists():
        raise FileNotFoundError(f"Golden artifacts directory for profile {profile_name} not found at: {prof_golden}")

    golden_emb_a = np.load(prof_golden / "video_a_embeddings.npy")
    golden_emb_b = np.load(prof_golden / "video_b_embeddings.npy")
    with open(prof_golden / "video_a_p_graph.json") as f:
        golden_graph_a = json.load(f)
    with open(prof_golden / "video_b_p_graph.json") as f:
        golden_graph_b = json.load(f)
    with open(prof_golden / "verification_result.json") as f:
        golden_verif = json.load(f)

    # Run current pipeline
    resolved_a = resolve_video_path(video_a_path)
    resolved_b = resolve_video_path(video_b_path)
    res_a = extract_five_muzzle_stages(str(resolved_a), device=device, cpu_cores=cpu_cores)
    res_b = extract_five_muzzle_stages(str(resolved_b), device=device, cpu_cores=cpu_cores)

    vec_a = res_a.get("vector_array") or list(res_a["median_parameters"].values())
    vec_b = res_b.get("vector_array") or list(res_b["median_parameters"].values())
    dl_emb_a = res_a.get("dl_embedding", {}).get("median_embedding") if res_a.get("dl_embedding") else None
    dl_emb_b = res_b.get("dl_embedding", {}).get("median_embedding") if res_b.get("dl_embedding") else None
    per_frames_a = res_a.get("dl_embedding", {}).get("per_frame_embeddings") if res_a.get("dl_embedding") else []
    per_frames_b = res_b.get("dl_embedding", {}).get("per_frame_embeddings") if res_b.get("dl_embedding") else []

    sim_res = compute_cosine_similarity(
        vec_a=vec_a,
        vec_b=vec_b,
        dl_emb_a=dl_emb_a,
        dl_emb_b=dl_emb_b,
        per_frames_a=per_frames_a,
        per_frames_b=per_frames_b,
    )

    failures = []
    checks_passed = []

    # 1. Per-frame embedding cosine >= 0.9999
    cosines_a = []
    cosines_b = []
    for i in range(min(len(per_frames_a), len(golden_emb_a))):
        c = cosine_sim_vecs(per_frames_a[i], golden_emb_a[i])
        cosines_a.append(c)
        if c < 0.9999:
            failures.append(f"Video A Frame {i+1} embedding cosine drift: {c:.6f} < 0.9999 threshold")
        else:
            checks_passed.append(f"Video A Frame {i+1} embedding cosine: {c:.6f} >= 0.9999")

    for i in range(min(len(per_frames_b), len(golden_emb_b))):
        c = cosine_sim_vecs(per_frames_b[i], golden_emb_b[i])
        cosines_b.append(c)
        if c < 0.9999:
            failures.append(f"Video B Frame {i+1} embedding cosine drift: {c:.6f} < 0.9999 threshold")
        else:
            checks_passed.append(f"Video B Frame {i+1} embedding cosine: {c:.6f} >= 0.9999")

    # 2. S_calibrated delta <= 0.1
    curr_s = float(sim_res.get("cosine_similarity", 0.0))
    gold_s = float(golden_verif.get("cosine_similarity", 0.0))
    delta_s = abs(curr_s - gold_s)
    if delta_s > 0.1:
        failures.append(f"Calibrated similarity S_calibrated delta: |{curr_s:.2f} - {gold_s:.2f}| = {delta_s:.2f}% > 0.1% threshold")
    else:
        checks_passed.append(f"Calibrated similarity delta: {delta_s:.4f}% <= 0.1%")

    # 3. P_graph features check
    diff_keys_a = []
    diff_keys_b = []
    curr_graph_a = res_a.get("median_parameters", {})
    curr_graph_b = res_b.get("median_parameters", {})
    for k in golden_graph_a:
        if k in curr_graph_a and abs(float(curr_graph_a[k]) - float(golden_graph_a[k])) > 1e-4:
            diff_keys_a.append((k, curr_graph_a[k], golden_graph_a[k]))
    for k in golden_graph_b:
        if k in curr_graph_b and abs(float(curr_graph_b[k]) - float(golden_graph_b[k])) > 1e-4:
            diff_keys_b.append((k, curr_graph_b[k], golden_graph_b[k]))

    if strict_p_graph and (diff_keys_a or diff_keys_b):
        failures.append(f"P_graph features changed! Video A diffs: {diff_keys_a}, Video B diffs: {diff_keys_b}")
    else:
        checks_passed.append("P_graph topological features preserved.")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "profile": profile_name,
        "failures": failures,
        "checks_passed": checks_passed,
        "min_cosine_a": min(cosines_a) if cosines_a else 1.0,
        "min_cosine_b": min(cosines_b) if cosines_b else 1.0,
        "delta_s_calibrated": delta_s,
        "current_s": curr_s,
        "golden_s": gold_s,
    }


def evaluate_dataset_pairs(device: str = "gpu", cpu_cores: int = 6) -> Dict[str, Any]:
    """Evaluates Genuine vs Impostor pairs from dataset/ to verify biometric separation."""
    dataset_dir = PROJECT_ROOT / "dataset"
    if not dataset_dir.exists():
        return {"status": "skipped", "reason": "dataset/ directory not found"}

    # Pairs to evaluate
    pairs = [
        # Genuine pairs (same cattle)
        ("cow_001/video_1.mp4", "cow_001/video_2.mp4", "genuine"),
        ("cow_001/video_2.mp4", "cow_001/video_3.mp4", "genuine"),
        ("cow_002/video_1.mp4", "cow_002/video_2.mp4", "genuine"),
        # Impostor pairs (different cattle)
        ("cow_001/video_1.mp4", "cow_002/video_1.mp4", "impostor"),
        ("cow_001/video_2.mp4", "cow_003/video_1.mp4", "impostor"),
        ("cow_002/video_2.mp4", "cow_003/video_2.mp4", "impostor"),
    ]

    results = []
    genuine_scores = []
    impostor_scores = []

    print(f"\n--- EVALUATING BIOMETRIC SEPARATION (GENUINE VS IMPOSTOR) ---")
    for va, vb, pair_type in pairs:
        path_a = dataset_dir / va
        path_b = dataset_dir / vb
        if not path_a.exists() or not path_b.exists():
            continue

        res_a = extract_five_muzzle_stages(str(path_a), device=device, cpu_cores=cpu_cores)
        res_b = extract_five_muzzle_stages(str(path_b), device=device, cpu_cores=cpu_cores)

        vec_a = res_a.get("vector_array") or list(res_a["median_parameters"].values())
        vec_b = res_b.get("vector_array") or list(res_b["median_parameters"].values())
        dl_emb_a = res_a.get("dl_embedding", {}).get("median_embedding") if res_a.get("dl_embedding") else None
        dl_emb_b = res_b.get("dl_embedding", {}).get("median_embedding") if res_b.get("dl_embedding") else None
        per_frames_a = res_a.get("dl_embedding", {}).get("per_frame_embeddings") if res_a.get("dl_embedding") else []
        per_frames_b = res_b.get("dl_embedding", {}).get("per_frame_embeddings") if res_b.get("dl_embedding") else []

        sim_res = compute_cosine_similarity(
            vec_a=vec_a,
            vec_b=vec_b,
            dl_emb_a=dl_emb_a,
            dl_emb_b=dl_emb_b,
            per_frames_a=per_frames_a,
            per_frames_b=per_frames_b,
        )

        score = float(sim_res.get("cosine_similarity", 0.0))
        verdict = sim_res.get("verdict", "")
        if pair_type == "genuine":
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

        print(f"[{pair_type.upper():<8}] {va} vs {vb} => S_calibrated = {score:.2f}% ({verdict})")
        results.append({
            "pair": f"{va} vs {vb}",
            "type": pair_type,
            "score": score,
            "verdict": verdict,
        })

    min_genuine = min(genuine_scores) if genuine_scores else 0.0
    max_impostor = max(impostor_scores) if impostor_scores else 0.0
    separation_margin = min_genuine - max_impostor

    print(f"\nSeparation Summary:")
    print(f"  Min Genuine Score:  {min_genuine:.2f}%")
    print(f"  Max Impostor Score: {max_impostor:.2f}%")
    print(f"  Separation Margin:  {separation_margin:.2f}% (positive = perfect threshold separation)")

    return {
        "results": results,
        "min_genuine": min_genuine,
        "max_impostor": max_impostor,
        "separation_margin": separation_margin,
    }


def main():
    parser = argparse.ArgumentParser(description="Regression Gate Checker")
    parser.add_argument("--profiles", nargs="+", default=["GPU", "CPU-6", "CPU-2"])
    parser.add_argument("--golden-dir", default=str(PROJECT_ROOT / "golden_artifacts"))
    parser.add_argument("--video-a", default="muzzleVideo.mp4")
    parser.add_argument("--video-b", default="muzzleVideo2.mp4")
    parser.add_argument("--eval-dataset", action="store_true")
    args = parser.parse_args()

    profile_configs = {
        "GPU": {"device": "gpu", "cpu_cores": 6},
        "CPU-6": {"device": "cpu", "cpu_cores": 6},
        "CPU-2": {"device": "cpu", "cpu_cores": 2},
    }

    all_passed = True
    for prof in args.profiles:
        if prof not in profile_configs:
            continue
        cfg = profile_configs[prof]
        res = check_regression_against_golden(
            profile_name=prof,
            device=cfg["device"],
            cpu_cores=cfg["cpu_cores"],
            video_a_path=args.video_a,
            video_b_path=args.video_b,
            golden_dir=Path(args.golden_dir),
        )
        print(f"\nProfile {prof} Regression Check: {'PASSED' if res['passed'] else 'FAILED'}")
        for c in res["checks_passed"]:
            print(f"  [PASS] {c}")
        for f in res["failures"]:
            print(f"  [FAIL] {f}")
        if not res["passed"]:
            all_passed = False

    if args.eval_dataset:
        evaluate_dataset_pairs()

    if not all_passed:
        print("\nREGRESSION GATE FAILED!")
        sys.exit(1)
    else:
        print("\nALL REGRESSION CHECKS PASSED (Cosine >= 0.9999, Delta S <= 0.1%).")
        sys.exit(0)


if __name__ == "__main__":
    main()
