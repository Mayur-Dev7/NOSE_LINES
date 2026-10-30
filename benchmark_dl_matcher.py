#!/usr/bin/env python3
"""
benchmark_dl_matcher.py
=======================
Verification and Benchmarking Script:
1. Verifies simultaneous TF and PyTorch operation.
2. Benchmarks inference latency (ms/frame) for ConvNeXt-Tiny and ResNet-50 on a 448x448 muzzle frame.
3. Tests 5-frame temporal median embedding aggregation and cosine matching.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure TF compatibility
os.environ["MRCNN_NVIDIA_CONFIGURED"] = "1"

print("=" * 65)
print("1. VERIFYING ENVIRONMENT & SIMULTANEOUS LIBRARY COMPATIBILITY")
print("=" * 65)

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
print("[OK] dl_muzzle_embedder loaded successfully.")

# Check for a sample image from output or create synthetic
sample_frame_path = PROJECT_ROOT / "output" / "video_five_frames"
found_clahe = None
for p in sample_frame_path.glob("**/frame_*_02_enhanced_ridge.jpg"):
    found_clahe = str(p)
    break

if found_clahe and os.path.exists(found_clahe):
    import cv2
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

print("\n" + "=" * 65)
print("2. INFERENCE LATENCY BENCHMARK (448x448 Frame, Warmup + 10 runs)")
print("=" * 65)

device_to_test = "cpu"
print(f"Device for benchmark: {device_to_test.upper()}")

results = {}

for backbone in ["convnext_tiny", "resnet50"]:
    model = get_muzzle_embedder(backbone=backbone, device=device_to_test)
    model.eval()

    # Warm-up run
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

print("\n" + "=" * 65)
print("3. TEMPORAL 5-FRAME EMBEDDING AGGREGATION & MATCHING TEST")
print("=" * 65)

# Simulate 5 frames for Video A
sim_frames_a = []
for i in range(5):
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
    sim_frames_a.append(emb)

# Simulate 5 frames for Video B (same animal with slight variance)
sim_frames_b_same = []
for i in range(5):
    noise = np.random.normal(0, 10, clahe_gray.shape).astype(np.uint8)
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
    sim_frames_b_same.append(emb)

# Simulate 5 frames for Video C (completely different random animal)
sim_frames_c_diff = []
for i in range(5):
    rand_clahe = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
    rand_groove = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
    rand_skel = (np.random.rand(448, 448) > 0.95).astype(np.uint8) * 255
    emb, _ = extract_frame_embedding(
        model=get_muzzle_embedder(backbone="convnext_tiny", device=device_to_test),
        clahe_gray=rand_clahe,
        groove_response=rand_groove,
        skeleton=rand_skel,
        mask=mask,
        input_mode="composite_3ch",
        device=device_to_test,
    )
    sim_frames_c_diff.append(emb)

median_a, intra_a = aggregate_temporal_embeddings(sim_frames_a)
median_b, intra_b = aggregate_temporal_embeddings(sim_frames_b_same)
median_c, intra_c = aggregate_temporal_embeddings(sim_frames_c_diff)

print(f"Video A: Median Norm={np.linalg.norm(median_a):.4f}, Intra-Consistency={intra_a:.2f}%")
print(f"Video B: Median Norm={np.linalg.norm(median_b):.4f}, Intra-Consistency={intra_b:.2f}%")
print(f"Video C: Median Norm={np.linalg.norm(median_c):.4f}, Intra-Consistency={intra_c:.2f}%")

match_same = compute_dl_cosine_similarity(median_a, median_b, sim_frames_a, sim_frames_b_same)
match_diff = compute_dl_cosine_similarity(median_a, median_c, sim_frames_a, sim_frames_c_diff)

print(f"\n[Genuine Pair: Video A vs Video B (Same Animal)]")
print(f"  Cosine Similarity: {match_same['dl_similarity_pct']:.2f}%")
print(f"  Euclidean L2 Dist: {match_same['l2_distance']:.4f}")
print(f"  Verdict:           {match_same['verdict']}")
print(f"  Cross Matrix Stats: {match_same['cross_matrix_stats']}")

print(f"\n[Impostor Pair: Video A vs Video C (Different Animals)]")
print(f"  Cosine Similarity: {match_diff['dl_similarity_pct']:.2f}%")
print(f"  Euclidean L2 Dist: {match_diff['l2_distance']:.4f}")
print(f"  Verdict:           {match_diff['verdict']}")
print(f"  Cross Matrix Stats: {match_diff['cross_matrix_stats']}")

print("\n" + "=" * 65)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 65)
