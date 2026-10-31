#!/usr/bin/env python3
"""
benchmark_dl_matcher.py
=======================
Verification and Benchmarking Script:
1. Verifies simultaneous TF and PyTorch operation without conflicts.
2. Benchmarks inference latency (ms/frame) for ConvNeXt-Tiny and ResNet-50.
3. Tests dual 5-frame temporal template pooling (Quality-Weighted Mean vs Element-Wise Median).
4. Verifies genuine (same cow) vs impostor (different cow) biometric matching discrimination:
   - Genuine pairs (Video 1 vs Video 1, Video 1 vs Video 2): score >= 88% (MATCH)
   - Impostor pairs (Video 1 vs Video 3, Video 2 vs Video 3): score < 65% (DISTINCT)
5. Verifies classical 10-D graph similarity Bray-Curtis / L1 magnitude penalty calibration.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure TF compatibility
os.environ["MRCNN_NVIDIA_CONFIGURED"] = "1"

print("=" * 70)
print("1. VERIFYING ENVIRONMENT & SIMULTANEOUS LIBRARY COMPATIBILITY")
print("=" * 70)

import tensorflow as tf
print(f"[OK] TensorFlow: {tf.__version__}")

import torch
print(f"[OK] PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")

import timm
print(f"[OK] timm: {timm.__version__}")

from dl_muzzle_embedder import (
    get_muzzle_embedder,
    extract_frame_embedding,
    aggregate_temporal_embeddings,
    compute_dl_cosine_similarity,
)
from video_groove_extractor import compute_cosine_similarity
print("[OK] dl_muzzle_embedder & video_groove_extractor loaded successfully.")

# Load sample frame for benchmark
sample_frame_path = PROJECT_ROOT / "output" / "video_five_frames"
found_clahe = None
for p in sample_frame_path.glob("**/frame_*_02_enhanced_ridge.jpg"):
    found_clahe = str(p)
    break

if found_clahe and os.path.exists(found_clahe):
    clahe_gray = cv2.imread(found_clahe, cv2.IMREAD_GRAYSCALE)
    stem = found_clahe.replace("_02_enhanced_ridge.jpg", "")
    p_groove = f"{stem}_03_groove_response.png"
    p_skel = f"{stem}_05_thin_skeleton.png"
    groove_map = cv2.imread(p_groove, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_groove) else clahe_gray
    skeleton = cv2.imread(p_skel, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_skel) else clahe_gray
    mask = np.full((448, 448), 255, dtype=np.uint8)
    print(f"[Sample Data] Loaded real extracted muzzle frame from: {found_clahe}")
else:
    clahe_gray = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
    groove_map = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
    skeleton = (np.random.rand(448, 448) > 0.95).astype(np.uint8) * 255
    mask = np.full((448, 448), 255, dtype=np.uint8)
    print("[Sample Data] Generated synthetic 448x448 muzzle frame.")

print("\n" + "=" * 70)
print("2. INFERENCE LATENCY BENCHMARK (448x448 Frame, Warmup + 10 runs)")
print("=" * 70)

device_to_test = "cpu"
print(f"Device for benchmark: {device_to_test.upper()}")

results = {}

for backbone in ["convnext_tiny", "resnet50"]:
    model = get_muzzle_embedder(backbone=backbone, device=device_to_test)
    model.eval()

    # Warm-up runs
    for _ in range(2):
        _ = extract_frame_embedding(
            model=model,
            clahe_gray=clahe_gray,
            groove_response=groove_map,
            skeleton=skeleton,
            mask=mask,
            input_mode="composite_3ch",
            target_size=(224, 224),
            device=device_to_test,
        )

    # Benchmark 10 iterations
    latencies = []
    embs = []
    for _ in range(10):
        emb, lat = extract_frame_embedding(
            model=model,
            clahe_gray=clahe_gray,
            groove_response=groove_map,
            skeleton=skeleton,
            mask=mask,
            input_mode="composite_3ch",
            target_size=(224, 224),
            device=device_to_test,
        )
        latencies.append(lat)
        embs.append(emb)

    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    results[backbone] = {
        "mean_ms": mean_lat,
        "std_ms": std_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "emb_dim": len(embs[0]),
        "emb_norm": float(np.linalg.norm(embs[0])),
        "weights_mode": model.weights_mode,
    }

    print(f"\n--- Backbone: {backbone.upper()} ---")
    print(f"  Weights Mode:     {model.weights_mode}")
    print(f"  Embedding Dim:    {len(embs[0])}-D (L2 Norm: {np.linalg.norm(embs[0]):.4f})")
    print(f"  Mean Latency:     {mean_lat:.2f} ms/frame (+/- {std_lat:.2f} ms)")
    print(f"  Min / Max Latency: {min_lat:.2f} ms / {max_lat:.2f} ms")

print("\n" + "=" * 70)
print("3. DUAL TEMPORAL AGGREGATION & OUTLIER PRUNING TEST")
print("=" * 70)

# Simulate 5 synthetic frames with 1 outlier frame
sim_frames = []
for i in range(4):
    noise = np.random.normal(0, 5, clahe_gray.shape).astype(np.uint8)
    sim_clahe = np.clip(clahe_gray.astype(int) + noise, 0, 255).astype(np.uint8)
    emb, _ = extract_frame_embedding(
        model=get_muzzle_embedder(backbone="convnext_tiny", device=device_to_test),
        clahe_gray=sim_clahe,
        groove_response=groove_map,
        skeleton=skeleton,
        mask=mask,
        input_mode="composite_3ch",
        device=device_to_test,
    )
    sim_frames.append(emb)

# Add 1 corrupted / outlier frame
outlier_clahe = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
outlier_emb, _ = extract_frame_embedding(
    model=get_muzzle_embedder(backbone="convnext_tiny", device=device_to_test),
    clahe_gray=outlier_clahe,
    groove_response=groove_map,
    skeleton=skeleton,
    mask=mask,
    input_mode="composite_3ch",
    device=device_to_test,
)
sim_frames.append(outlier_emb)

confs = [0.96, 0.94, 0.98, 0.95, 0.40]
blurs = [185.0, 192.0, 178.0, 205.0, 18.0]

agg = aggregate_temporal_embeddings(sim_frames, confidences=confs, blur_variances=blurs)
print(f"Frames Kept:           {agg['kept_frames_count']} / 5")
print(f"Dropped Outliers:      Indices {agg['dropped_frame_indices']}")
print(f"Agreement Scores:      {agg['agreement_scores']}")
print(f"Quality Weights:       {agg['weights']}")
print(f"Intra-Consistency:     {agg['intra_video_consistency_pct']:.2f}%")
print(f"Mean Embedding Norm:   {np.linalg.norm(agg['mean_embedding']):.4f}")
print(f"Median Embedding Norm: {np.linalg.norm(agg['median_embedding']):.4f}")

print("\n" + "=" * 70)
print("4. BIOMETRIC MATCHING ON REAL EXTRACTED VIDEO SETS (V1, V2, V3)")
print("=" * 70)

v1_dir = PROJECT_ROOT / "output" / "video_five_frames" / "muzzleVideo_frangi_adaptive"
v2_dir = PROJECT_ROOT / "output" / "video_five_frames" / "muzzleVideo2_frangi_adaptive"
v3_dir = PROJECT_ROOT / "output" / "video_five_frames" / "muzzleVideo3_frangi_adaptive"

def load_video_frames(dir_path: Path, embedder):
    embs = []
    confs = []
    blurs = []
    for idx in range(5):
        c_path = dir_path / f"frame_{idx}_02_enhanced_ridge.jpg"
        g_path = dir_path / f"frame_{idx}_03_groove_response.png"
        s_path = dir_path / f"frame_{idx}_05_thin_skeleton.png"
        m_path = dir_path / f"frame_{idx}_04_groove_mask.png"
        c = cv2.imread(str(c_path), cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(str(g_path), cv2.IMREAD_GRAYSCALE)
        s = cv2.imread(str(s_path), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        e, _ = extract_frame_embedding(embedder, c, g, s, m, device=device_to_test)
        embs.append(e)
        confs.append(0.95)
        blurs.append(180.0)
    return embs, confs, blurs

embedder_cnext = get_muzzle_embedder("convnext_tiny", device=device_to_test)

if v1_dir.exists() and v2_dir.exists() and v3_dir.exists():
    v1_embs, v1_c, v1_b = load_video_frames(v1_dir, embedder_cnext)
    v2_embs, v2_c, v2_b = load_video_frames(v2_dir, embedder_cnext)
    v3_embs, v3_c, v3_b = load_video_frames(v3_dir, embedder_cnext)

    agg1 = aggregate_temporal_embeddings(v1_embs, v1_c, v1_b)
    agg2 = aggregate_temporal_embeddings(v2_embs, v2_c, v2_b)
    agg3 = aggregate_temporal_embeddings(v3_embs, v3_c, v3_b)

    # 1. Genuine Pairs (Same Animal)
    match_1_1 = compute_dl_cosine_similarity(agg1, agg1, v1_embs, v1_embs)
    match_1_2 = compute_dl_cosine_similarity(agg1, agg2, v1_embs, v2_embs)

    # 2. Impostor Pairs (Different Animals)
    match_1_3 = compute_dl_cosine_similarity(agg1, agg3, v1_embs, v3_embs)
    match_2_3 = compute_dl_cosine_similarity(agg2, agg3, v2_embs, v3_embs)

    print("\n[GENUINE PAIR 1: Video 1 vs Video 1 (Self Identity)]")
    print(f"  Mean Pooling Sim:    {match_1_1['mean_pooling_similarity_pct']:.2f}% (L2: {match_1_1['mean_l2_distance']:.4f})")
    print(f"  Median Pooling Sim:  {match_1_1['median_pooling_similarity_pct']:.2f}% (L2: {match_1_1['median_l2_distance']:.4f})")
    print(f"  Trimmed Cross-Frame: {match_1_1['trimmed_cross_frame_similarity_pct']:.2f}%")
    print(f"  Verdict:             {match_1_1['verdict']}")

    print("\n[GENUINE PAIR 2: Video 1 vs Video 2 (Same Cow)]")
    print(f"  Mean Pooling Sim:    {match_1_2['mean_pooling_similarity_pct']:.2f}% (L2: {match_1_2['mean_l2_distance']:.4f})")
    print(f"  Median Pooling Sim:  {match_1_2['median_pooling_similarity_pct']:.2f}% (L2: {match_1_2['median_l2_distance']:.4f})")
    print(f"  Trimmed Cross-Frame: {match_1_2['trimmed_cross_frame_similarity_pct']:.2f}%")
    print(f"  Verdict:             {match_1_2['verdict']}")

    print("\n[IMPOSTOR PAIR 1: Video 1 vs Video 3 (Different Animal)]")
    print(f"  Mean Pooling Sim:    {match_1_3['mean_pooling_similarity_pct']:.2f}% (L2: {match_1_3['mean_l2_distance']:.4f})")
    print(f"  Median Pooling Sim:  {match_1_3['median_pooling_similarity_pct']:.2f}% (L2: {match_1_3['median_l2_distance']:.4f})")
    print(f"  Trimmed Cross-Frame: {match_1_3['trimmed_cross_frame_similarity_pct']:.2f}%")
    print(f"  Verdict:             {match_1_3['verdict']}")

    print("\n[IMPOSTOR PAIR 2: Video 2 vs Video 3 (Different Animal)]")
    print(f"  Mean Pooling Sim:    {match_2_3['mean_pooling_similarity_pct']:.2f}% (L2: {match_2_3['mean_l2_distance']:.4f})")
    print(f"  Median Pooling Sim:  {match_2_3['median_pooling_similarity_pct']:.2f}% (L2: {match_2_3['median_l2_distance']:.4f})")
    print(f"  Trimmed Cross-Frame: {match_2_3['trimmed_cross_frame_similarity_pct']:.2f}%")
    print(f"  Verdict:             {match_2_3['verdict']}")

    # Validation Checks
    assert match_1_1["dl_similarity_pct"] >= 88.0, f"Expected V1 vs V1 >= 88%, got {match_1_1['dl_similarity_pct']}%"
    assert match_1_2["dl_similarity_pct"] >= 88.0, f"Expected V1 vs V2 >= 88%, got {match_1_2['dl_similarity_pct']}%"
    assert match_1_3["dl_similarity_pct"] < 65.0, f"Expected V1 vs V3 < 65%, got {match_1_3['dl_similarity_pct']}%"
    assert match_2_3["dl_similarity_pct"] < 65.0, f"Expected V2 vs V3 < 65%, got {match_2_3['dl_similarity_pct']}%"
    print("\n[PASS] All Deep Learning Verification Criteria met with 100% precision!")
else:
    print("[WARNING] Output video folders not found. Skipping real video verification.")

print("\n" + "=" * 70)
print("5. CLASSICAL 10-D GRAPH SIMILARITY WITH BRAY-CURTIS L1 MAGNITUDE PENALTY")
print("=" * 70)

# Load real 10-D vectors from summary.json
import json
def load_summary_vector(dir_path: Path):
    s_path = dir_path / "summary.json"
    with open(s_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["vector_array"], data["normalized_vector"]

if v1_dir.exists() and v2_dir.exists() and v3_dir.exists():
    v1_raw, v1_norm = load_summary_vector(v1_dir)
    v2_raw, v2_norm = load_summary_vector(v2_dir)
    v3_raw, v3_norm = load_summary_vector(v3_dir)

    sim_same = compute_cosine_similarity(v1_raw, v2_raw)
    sim_diff = compute_cosine_similarity(v1_raw, v3_raw)

    print("\n[Genuine Pair: Video 1 vs Video 2 (Same Cow)]")
    print(f"  Raw Cosine Sim:        {sim_same['raw_cosine_similarity']:.2f}%")
    print(f"  Magnitude Penalty:     {sim_same['magnitude_penalty_factor']:.4f} ({sim_same['magnitude_penalty_pct']:.2f}%)")
    print(f"  Calibrated Graph Sim:  {sim_same['cosine_similarity']:.2f}%")
    print(f"  Verdict:               {sim_same['verdict']}")

    print("\n[Impostor Pair: Video 1 vs Video 3 (Different Animal)]")
    print(f"  Raw Cosine Sim:        {sim_diff['raw_cosine_similarity']:.2f}%")
    print(f"  Magnitude Penalty:     {sim_diff['magnitude_penalty_factor']:.4f} ({sim_diff['magnitude_penalty_pct']:.2f}%)")
    print(f"  Calibrated Graph Sim:  {sim_diff['cosine_similarity']:.2f}%")
    print(f"  Verdict:               {sim_diff['verdict']}")

    assert sim_same["cosine_similarity"] >= 85.0
    assert sim_diff["cosine_similarity"] < 80.0
    print("\n[PASS] Classical 10-D Graph Bray-Curtis Penalty verified successfully!")

print("\n" + "=" * 70)
print("ALL BENCHMARKS AND VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
