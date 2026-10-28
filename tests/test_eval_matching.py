"""
Unit and Integration Tests for Evaluation Harness and ROC/EER Biometric Calibration
===================================================================================
Tests tools/eval_matching.py and biometric calibration artifacts.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from tools.eval_matching import (
    scan_dataset,
    compute_distribution_stats,
    compute_roc_eer,
    generate_synthetic_demo_dataset,
    PROJECT_ROOT,
)
from muzzle_matcher import (
    compare_two_frames_biometric,
    load_matcher_config,
    CANONICAL_SIZE,
)


def test_scan_dataset_empty_and_populated():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Empty
        assert scan_dataset(tmp_path) == {}

        # Non-directory
        assert scan_dataset(tmp_path / "nonexistent") == {}

        # Populated with cows and videos
        cow1 = tmp_path / "cow_001"
        cow2 = tmp_path / "cow_002"
        cow1.mkdir()
        cow2.mkdir()
        (cow1 / "clip1.mp4").touch()
        (cow1 / "clip2.mp4").touch()
        (cow2 / "clip1.mp4").touch()
        (cow2 / "ignore.txt").touch()

        cows = scan_dataset(tmp_path)
        assert len(cows) == 2
        assert len(cows["cow_001"]) == 2
        assert len(cows["cow_002"]) == 1
        assert cows["cow_001"][0].name == "clip1.mp4"


def test_compute_distribution_stats():
    # Empty
    empty_stats = compute_distribution_stats([])
    assert empty_stats["count"] == 0
    assert empty_stats["mean"] == 0.0

    # Populated
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = compute_distribution_stats(scores)
    assert stats["count"] == 5
    assert np.isclose(stats["mean"], 0.3, atol=1e-3)
    assert np.isclose(stats["median"], 0.3, atol=1e-3)
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5


def test_compute_roc_eer_separable():
    genuine = [0.75, 0.82, 0.88, 0.94, 0.98]
    impostor = [0.12, 0.18, 0.22, 0.28, 0.35]

    res = compute_roc_eer(genuine, impostor)
    assert res["eer"] == 0.0
    assert np.isclose(res["roc_auc"], 1.0, atol=1e-2)
    assert 0.35 <= res["eer_threshold"] <= 0.75
    assert res["tar_at_eer"] == 1.0
    assert res["far_at_eer"] == 0.0
    assert len(res["threshold_table"]) > 0


def test_compute_roc_eer_overlapping():
    genuine = [0.30, 0.50, 0.60, 0.80]
    impostor = [0.20, 0.40, 0.55, 0.70]

    res = compute_roc_eer(genuine, impostor)
    assert 0.0 <= res["eer"] <= 0.50
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["eer_threshold"] <= 1.0


def test_compute_roc_eer_empty():
    res = compute_roc_eer([], [])
    assert res["eer"] == 0.0
    assert res["roc_auc"] == 0.5


def test_synthetic_dataset_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cows = generate_synthetic_demo_dataset(
            dataset_dir=tmp_path,
            num_cows=2,
            videos_per_cow=2,
            frames_per_video=5,
        )
        assert len(cows) == 2
        for cow_id, vids in cows.items():
            assert len(vids) == 2
            for v in vids:
                assert v.exists()
                assert v.stat().st_size > 500


def test_compare_two_frames_biometric():
    # Synthetic frame test
    gray1 = np.full(CANONICAL_SIZE, 128, dtype=np.uint8)
    gray1[100:120, :] = 220
    gray1[:, 200:220] = 220

    gray2 = gray1.copy()

    res = compare_two_frames_biometric(gray1, gray2)
    assert "composite_score" in res
    assert "sift_inlier_count" in res
    assert "minutiae_match_fraction" in res
    assert res["composite_score"] >= 0.50


def test_matcher_config_file_validity():
    cfg_path = PROJECT_ROOT / "matcher_config.json"
    assert cfg_path.exists(), f"matcher_config.json missing at {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("calibrated") is True
    assert "calibrated_at" in data
    assert isinstance(data.get("threshold"), (int, float))
    assert 0.0 <= data["threshold"] <= 1.0
    assert isinstance(data.get("eer"), (int, float))
    assert isinstance(data.get("roc_auc"), (int, float))
    assert data["roc_auc"] >= 0.80
    assert isinstance(data.get("min_sift_inliers"), int)
    assert isinstance(data.get("min_inlier_ratio"), float)
    assert isinstance(data.get("min_minutiae_fraction"), float)
    assert data.get("decision_metric") == "composite_score"
    assert "summary" in data

    # Verify load_matcher_config loads it
    loaded = load_matcher_config(cfg_path)
    assert loaded["calibrated"] is True
    assert loaded["threshold"] == data["threshold"]


def test_eval_matching_cli_help():
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
        str(PROJECT_ROOT / "tools" / "eval_matching.py"),
        "--help",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "Cattle Muzzle Biometric Evaluation Harness" in res.stdout
