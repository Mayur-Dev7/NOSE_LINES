#!/usr/bin/env python3
"""
test_api_endpoints.py
=====================
Direct integration tests for server endpoints with Deep Learning ArcFace / GeM Matching:
- Tests compute_cosine_similarity_endpoint
- Tests request parsing & serialization
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import (
    CosineSimilarityRequest,
    compute_cosine_similarity_endpoint,
    ExtractVideoFiveFramesRequest,
    _spatial_cache_key,
    get_embedding_gallery_presets_endpoint,
    match_one_to_n_endpoint,
    OneToNMatchRequest,
)

async def test_cosine_similarity():
    vec_a = [25000.0, 8000.0, 28.5, 320.0, 210.0, 85.0, 16.2, 75.0, 0.045, -45.0]
    vec_b = [24800.0, 7950.0, 28.1, 315.0, 208.0, 84.0, 16.5, 74.0, 0.044, -44.0]

    rng = np.random.RandomState(42)
    raw_a = rng.randn(512).astype(float)
    emb_a = (raw_a / np.linalg.norm(raw_a)).tolist()

    raw_b = raw_a + rng.normal(0, 0.05, 512)
    emb_b = (raw_b / np.linalg.norm(raw_b)).tolist()

    # 5 frames simulated for A and B
    per_frames_a = [(raw_a + rng.normal(0, 0.02, 512)).tolist() for _ in range(5)]
    per_frames_b = [(raw_b + rng.normal(0, 0.02, 512)).tolist() for _ in range(5)]

    req = CosineSimilarityRequest(
        vector_a=vec_a,
        vector_b=vec_b,
        dl_emb_a=emb_a,
        dl_emb_b=emb_b,
        per_frames_a=per_frames_a,
        per_frames_b=per_frames_b,
    )

    res = await compute_cosine_similarity_endpoint(req)

    assert "cosine_similarity" in res
    assert "dl_similarity_pct" in res
    assert "l2_distance" in res
    assert "dl_verdict" in res
    assert "cross_matrix" in res
    assert "cross_matrix_stats" in res
    assert len(res["cross_matrix"]) == 5
    assert res["dl_similarity_pct"] > 95.0, f"Expected >95%, got {res['dl_similarity_pct']}"
    print(f"[PASS] Cosine Similarity Endpoint:")
    print(f"       Graph Cosine: {res['cosine_similarity']}%")
    print(f"       DL ArcFace/GeM Cosine: {res['dl_similarity_pct']}%")
    print(f"       Euclidean L2 Distance: {res['l2_distance']}")
    print(f"       Verdict: {res['dl_verdict']}")
    print(f"       Cross-Matrix Stats: {res['cross_matrix_stats']}")


def test_extraction_request_model():
    req = ExtractVideoFiveFramesRequest(
        video_path="/home/adis_mayur/F_CODES/muzzleVideo.mp4",
        dl_backbone="convnext_tiny",
        dl_input_mode="composite_3ch",
    )
    assert req.dl_backbone == "convnext_tiny"
    assert req.dl_input_mode == "composite_3ch"
    print(f"[PASS] ExtractVideoFiveFramesRequest model validated: backbone={req.dl_backbone}, mode={req.dl_input_mode}")


async def test_gallery_and_1_to_n_search():
    presets = await get_embedding_gallery_presets_endpoint()
    assert "gallery" in presets
    assert len(presets["gallery"]) >= 5
    assert "sample_cow_a" in presets
    print(f"[PASS] Gallery Presets Endpoint returned {len(presets['gallery'])} registered cattle")

    # Match Sample Cow A against Full Gallery
    q_emb = presets["sample_cow_a"]["embedding"]
    req = OneToNMatchRequest(query_embedding=q_emb, threshold=88.0)
    res = await match_one_to_n_endpoint(req)

    assert res["success"] is True
    assert res["verdict_is_match"] is True
    assert "Cattle A" in res["top_match"]["name"]
    assert res["top_match"]["similarity_pct"] >= 88.0
    print(f"[PASS] 1-to-N Match Endpoint (Full Gallery):")
    print(f"       Verdict: {res['verdict']}")
    print(f"       Top Match: {res['top_match']['name']} ({res['top_match']['similarity_pct']}%)")
    print(f"       Euclidean L2: {res['top_match']['l2_distance']}")
    print(f"       Lead Margin Ahead: +{res['margin_ahead_pct']}%")

    # Test cross-video re-identification (querying Cattle A Profile 1 against gallery of alternative takes)
    gallery_cross = [c for c in presets["gallery"] if "Profile 1" not in c["name"]]
    req_cross = OneToNMatchRequest(query_embedding=q_emb, gallery=gallery_cross, threshold=88.0)
    res_cross = await match_one_to_n_endpoint(req_cross)
    assert res_cross["success"] is True
    assert res_cross["verdict_is_match"] is True
    assert "Cattle A (Profile 2" in res_cross["top_match"]["name"]
    assert res_cross["top_match"]["similarity_pct"] >= 88.0
    assert res_cross["margin_ahead_pct"] > 40.0
    print(f"[PASS] 1-to-N Cross-Video Re-identification:")
    print(f"       Top Match: {res_cross['top_match']['name']} ({res_cross['top_match']['similarity_pct']}%)")
    print(f"       Lead Margin Ahead of Cattle B/C/D/E: +{res_cross['margin_ahead_pct']}%")


if __name__ == "__main__":
    print("=== RUNNING FASTAPI ENDPOINT INTEGRATION TESTS ===")
    test_extraction_request_model()
    asyncio.run(test_cosine_similarity())
    asyncio.run(test_gallery_and_1_to_n_search())
    print("\n=== ALL INTEGRATION TESTS PASSED WITH 100% SUCCESS! ===")
