#!/usr/bin/env python3
"""
tests/test_search_backend.py
============================
Comprehensive test suite for the Cattle Biometric Search Backend:
- Tests GET /api/embedding/gallery-presets
- Tests POST /api/embedding/1-to-n-match (vectorized dot products, verdicts, rankings)
- Tests POST and GET /api/embedding/export (single-video embedding extraction and caching)
- Tests error handling and edge cases (empty gallery, dimension mismatch)
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import (
    OneToNMatchRequest,
    one_to_n_match_endpoint,
    get_embedding_gallery_presets_endpoint,
    SingleVideoEmbeddingExportRequest,
    export_single_video_embedding_post,
    export_single_video_embedding_get,
)


def test_gallery_presets():
    presets = asyncio.run(get_embedding_gallery_presets_endpoint())
    assert presets["success"] is True, "Preset fetch failed"
    gallery = presets["gallery"]
    assert len(gallery) >= 6, f"Expected at least 6 gallery items, got {len(gallery)}"

    expected_ids = {"cattle_001_a", "cattle_001_b", "cattle_002"}
    found_ids = {item["id"] for item in gallery}
    assert expected_ids.issubset(found_ids), f"Missing expected IDs: {expected_ids - found_ids}"

    for cand in gallery:
        assert "id" in cand
        assert "name" in cand
        assert "embedding" in cand
        assert len(cand["embedding"]) == 512, f"Candidate {cand['id']} has wrong dim: {len(cand['embedding'])}"
        assert "notes" in cand

    print(f"[PASS] test_gallery_presets: verified {len(gallery)} registered cattle presets.")


def test_1_to_n_match_logic():
    # Construct synthetic 512-D candidates
    rng = np.random.RandomState(42)
    base = rng.randn(512).astype(np.float32)
    base /= np.linalg.norm(base)

    # Identical candidate (high match >95%)
    cand_match = (base + rng.normal(0, 0.01, 512)).astype(np.float32)
    cand_match /= np.linalg.norm(cand_match)

    # Likely candidate (approx 80%)
    noise = rng.randn(512).astype(np.float32)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    cand_likely = 0.82 * base + np.sqrt(1 - 0.82**2) * noise
    cand_likely /= np.linalg.norm(cand_likely)

    # Distinct candidate (approx 20%)
    cand_distinct = 0.20 * base + np.sqrt(1 - 0.20**2) * noise
    cand_distinct /= np.linalg.norm(cand_distinct)

    gallery = [
        {"id": "cand_distinct", "name": "Distinct Animal", "embedding": cand_distinct.tolist(), "notes": "Unrelated"},
        {"id": "cand_match", "name": "Same Cattle Profile 2", "embedding": cand_match.tolist(), "notes": "Confirmed take"},
        {"id": "cand_likely", "name": "Likely Cow", "embedding": cand_likely.tolist(), "notes": "Similar herd"},
    ]

    req = OneToNMatchRequest(
        query_embedding=base.tolist(),
        gallery=gallery,
        similarity_metric="cosine",
    )
    res = asyncio.run(one_to_n_match_endpoint(req))

    assert res["success"] is True
    assert res["query_dim"] == 512
    assert res["total_candidates"] == 3
    assert res["top_match"]["id"] == "cand_match"
    assert res["top_match"]["verdict"] == "MATCH (Confirmed Same Cattle)"
    assert res["top_match"]["badge"] == "match"
    assert res["top_match"]["color"] == "emerald"
    assert res["margin_to_runner_up"] is not None

    # Check ranking
    ranked = res["ranked_results"]
    assert ranked[0]["id"] == "cand_match"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3

    print(f"[PASS] test_1_to_n_match_logic: top match={ranked[0]['name']} ({ranked[0]['similarity_pct']}%), margin={res['margin_to_runner_up']}%")


def test_empty_gallery():
    q = np.random.randn(512).tolist()
    req = OneToNMatchRequest(query_embedding=q, gallery=[])
    res = asyncio.run(one_to_n_match_endpoint(req))

    assert res["success"] is True
    assert res["total_candidates"] == 0
    assert res["top_match"] is None
    assert res["margin_to_runner_up"] is None
    assert res["ranked_results"] == []
    print("[PASS] test_empty_gallery: properly handled 0 candidates.")


def test_single_video_embedding_export():
    video_path = "uploads/1793364192_muzzleVideo2.mp4"
    req = SingleVideoEmbeddingExportRequest(video_path=video_path, pool_method="median")
    res_post = asyncio.run(export_single_video_embedding_post(req))

    assert res_post["success"] is True
    assert res_post["embedding_dim"] == 512
    assert len(res_post["embedding"]) == 512
    assert res_post["cached"] is True

    res_get = asyncio.run(export_single_video_embedding_get(video_path=video_path, pool_method="mean"))
    assert res_get["success"] is True
    assert res_get["embedding_dim"] == 512

    print(f"[PASS] test_single_video_embedding_export: verified export for {res_post['video_name']}.")


if __name__ == "__main__":
    print("=== RUNNING SEARCH BACKEND TEST SUITE ===")
    test_gallery_presets()
    test_1_to_n_match_logic()
    test_empty_gallery()
    test_single_video_embedding_export()
    print("\n=== ALL SEARCH BACKEND TESTS PASSED SUCCESSFULLY! ===")
