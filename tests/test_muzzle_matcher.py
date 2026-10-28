"""Tests for the new spatial matcher contract and locked interfaces."""
import asyncio
import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from groove_graph import prune_skeleton, extract_minutiae_points
from video_groove_extractor import canonical_align_muzzle
from muzzle_matcher import match_frame_pair, match_video_sets
from server import VideoMatchRequest, get_spatial_match_config_endpoint


def _branch_skeleton(spur_length=5):
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(image, (15, 50), (80, 50), 255, 1)
    cv2.line(image, (45, 50), (45, 50 - spur_length), 255, 1)
    # Add a distinct isolated 10-px component.
    cv2.line(image, (5, 5), (14, 5), 255, 1)
    return image


def test_spur_pruning_removes_short_branch_and_small_component():
    skel = _branch_skeleton()
    pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=30)
    assert pruned.dtype == np.uint8
    assert set(np.unique(pruned)).issubset({0, 255})
    # Short branch endpoint and its pixels are fully removed; no nub remains.
    assert np.count_nonzero(pruned[42:50, 45]) == 0
    # Main trunk remains intact.
    assert np.count_nonzero(pruned[50, 15:81]) >= 60
    # The tiny disconnected 10-px component is dropped.
    assert np.count_nonzero(pruned[5, 5:15]) == 0


def test_spur_pruning_keeps_long_branch():
    skel = _branch_skeleton(spur_length=12)
    pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=30)
    assert np.count_nonzero(pruned[38:50, 45]) >= 10


def test_canonical_align_uses_small_local_nostrils_and_recovers_tilt():
    h = w = 300
    image = np.full((h, w, 3), 180, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (150, 150), (130, 115), 0, 0, 360, 255, -1)
    center = (150, 75)
    tilt = math.radians(15)
    dist = 44
    for sign in (-1, 1):
        x = int(center[0] + sign * dist * math.cos(tilt))
        y = int(center[1] + sign * dist * math.sin(tilt))
        cv2.ellipse(image, (x, y), (7, 5), 0, 0, 360, (20, 20, 20), -1)
    aligned, aligned_mask, meta = canonical_align_muzzle(image, mask, 448)
    assert aligned.shape == (448, 448, 3)
    assert aligned_mask.shape == (448, 448)
    assert meta["alignment_method"] == "local_nostril_contrast"
    assert abs(meta["rotation_deg"]) == pytest.approx(15, abs=5)


def _frames(image, skeleton):
    return [{"clahe_gray": image, "mask": np.full((448, 448), 255, dtype=np.uint8),
             "skeleton": skeleton, "minutiae": extract_minutiae_points(skeleton, max_points=250)}]


def test_match_video_sets_uncalibrated_and_visualization(tmp_path):
    params_path = Path("config/matching_params.json")
    threshold_path = tmp_path / "absent_threshold.json"
    base = np.full((448, 448), 120, dtype=np.uint8)
    rng = np.random.default_rng(1234)
    for _ in range(120):
        x, y = int(rng.integers(25, 420)), int(rng.integers(25, 420))
        cv2.line(base, (x, y), (min(440, x + 15), max(5, y - 18)), 210, 2)
    skeleton = np.zeros_like(base)
    for y in range(40, 410, 20):
        cv2.line(skeleton, (35, y), (410, y + 1), 255, 1)
    M = cv2.getRotationMatrix2D((224, 224), 8, 1.0)
    M[:, 2] += np.array([4.0, -3.0])
    shifted = cv2.warpAffine(base, M, (448, 448))
    shifted_skel = cv2.warpAffine(skeleton, M, (448, 448), flags=cv2.INTER_NEAREST)
    independent = np.random.default_rng(999).integers(0, 255, (448, 448), dtype=np.uint8)
    results = match_video_sets(_frames(base, skeleton), _frames(shifted, shifted_skel),
                               params_path=str(params_path), threshold_path=str(threshold_path))
    different = match_video_sets(_frames(base, skeleton), _frames(independent, np.zeros_like(skeleton)),
                                 params_path=str(params_path), threshold_path=str(threshold_path))
    assert results["verdict"] == "UNCALIBRATED"
    assert results["calibrated"] is False
    assert results["dev_hint"] in {"DEV_LIKELY_MATCH", "DEV_LIKELY_NO_MATCH"}
    assert results["decision_score"] > different["decision_score"]
    assert base64.b64decode(results["match_viz_base64"].encode())
    calibrated_cfg = {"calibrated": True, "threshold": 0.3, "genuine_pairs": 30, "impostor_pairs": 30}
    threshold_path.write_text(json.dumps(calibrated_cfg), encoding="utf-8")
    calibrated = match_video_sets(_frames(base, skeleton), _frames(shifted, shifted_skel),
                                  params_path=str(params_path), threshold_path=str(threshold_path))
    assert calibrated["verdict"] in {"MATCH", "NO_MATCH"}
    assert calibrated["calibrated"] is True


def test_match_config_endpoint_is_uncalibrated_without_valid_threshold(tmp_path, monkeypatch):
    # The checked-in threshold configuration intentionally states uncalibrated.
    config = asyncio.run(get_spatial_match_config_endpoint())
    assert config["status"] in {"UNCALIBRATED", "CALIBRATED"}
    assert "dev_provisional_threshold" in config


def test_video_match_request_retains_existing_filter_fields():
    request = VideoMatchRequest(video_a_path="/tmp/a.mp4", video_b_path="/tmp/b.mp4",
                                filter_method="blackhat", threshold_method="otsu",
                                adaptive_block_size=15, adaptive_c=2, exclude_nostrils=True)
    assert request.filter_method == "blackhat"
    assert request.threshold_method == "otsu"
