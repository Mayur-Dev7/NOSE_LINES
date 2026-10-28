"""
Unit tests for SIFT/RANSAC and Minutiae Biometric Matcher & Video Match Endpoints
=================================================================================
Tests:
- Method A: SIFT Keypoint Matching with ORB Fallback + RANSAC
- Method B: Minutiae Matching from Topological Endpoints & Junctions
- 5x5 All-Pairs Comparison & Visualization Generation
- Matcher Config Loader
- FastAPI Match & Config Endpoints
"""

import os
import sys
import json
import asyncio
import numpy as np
import pytest
import cv2
from pathlib import Path

# Ensure project root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from muzzle_matcher import (
    match_sift_ransac,
    extract_minutiae_points,
    match_minutiae,
    match_muzzle_frame_sets,
    load_matcher_config,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_DUPLICATE_THRESHOLD,
    compute_image_quality,
    enhance_muzzle_image,
)
from server import (
    get_matcher_config_endpoint,
    match_video_endpoint,
    VideoMatchRequest,
)
from fastapi import HTTPException


@pytest.fixture
def synthetic_textured_images():
    """Generates a textured 448x448 muzzle image and a slightly shifted version."""
    img1 = np.zeros((448, 448), dtype=np.uint8)
    for i in range(30, 420, 20):
        cv2.line(img1, (i, 30), (i, 420), 200, 2)
        cv2.circle(img1, (i, i), 8, 255, -1)
        cv2.circle(img1, (448 - i, i), 6, 180, -1)

    # Shift by (dx=6, dy=-4)
    M = np.float32([[1, 0, 6], [0, 1, -4]])
    img2 = cv2.warpAffine(img1, M, (448, 448))
    return img1, img2


def test_existing_utilities_preserved():
    """Verify legacy exports compute_image_quality and enhance_muzzle_image remain intact."""
    test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    q = compute_image_quality(test_img)
    assert "composite" in q
    assert "sharpness" in q
    assert DEFAULT_DUPLICATE_THRESHOLD == 35

    enh_gray, up_bgr = enhance_muzzle_image(test_img, scale_factor=2)
    assert enh_gray.shape == (200, 200)
    assert up_bgr.shape == (200, 200, 3)


def test_match_sift_ransac_identical_and_shifted(synthetic_textured_images):
    """Verify Method A SIFT/RANSAC matches shifted pairs with high inliers."""
    img1, img2 = synthetic_textured_images
    res = match_sift_ransac(img1, img2)

    assert res["detector"] in ["sift", "orb"]
    assert res["inlier_count"] > 0
    assert res["inlier_ratio"] > 0.5
    assert 0.0 <= res["score_method_a"] <= 1.0
    assert res["transform_matrix"] is not None
    assert len(res["inliers_mask"]) == len(res["good_matches"])


def test_match_sift_ransac_blank_images():
    """Verify Method A handles blank images gracefully without crashing."""
    blank1 = np.zeros((448, 448), dtype=np.uint8)
    blank2 = np.zeros((448, 448), dtype=np.uint8)
    res = match_sift_ransac(blank1, blank2)

    assert res["inlier_count"] == 0
    assert res["score_method_a"] == 0.0
    assert res["good_matches_count"] == 0


def test_extract_minutiae_points():
    """Verify topological minutiae extraction from binary skeleton."""
    skel = np.zeros((200, 200), dtype=np.uint8)
    # T-junction and endpoints
    cv2.line(skel, (40, 100), (160, 100), 255, 1)
    cv2.line(skel, (100, 100), (100, 180), 255, 1)

    minutiae = extract_minutiae_points(skel)
    assert len(minutiae) >= 3
    types = [m["type"] for m in minutiae]
    assert "endpoint" in types
    for m in minutiae:
        assert "x" in m and "y" in m
        assert 0.0 <= m["orientation"] <= 360.0


def test_match_minutiae_ransac():
    """Verify Method B minutiae RANSAC alignment."""
    mins_a = [
        {"x": 50.0, "y": 60.0, "type": "endpoint", "orientation": 45.0},
        {"x": 120.0, "y": 80.0, "type": "junction", "orientation": 90.0},
        {"x": 200.0, "y": 150.0, "type": "endpoint", "orientation": 180.0},
        {"x": 250.0, "y": 280.0, "type": "junction", "orientation": 270.0},
        {"x": 300.0, "y": 100.0, "type": "endpoint", "orientation": 0.0},
    ]
    mins_b = [
        {"x": m["x"] + 10.0, "y": m["y"] - 5.0, "type": m["type"], "orientation": m["orientation"]}
        for m in mins_a
    ]

    res = match_minutiae(mins_a, mins_b, spatial_threshold=15.0, angular_threshold=35.0)
    assert res["matched_count"] >= 4
    assert res["matched_fraction"] >= 0.80
    assert res["score"] >= 0.80
    assert res["score_method_b"] >= 0.80
    assert res["transform_matrix"] is not None


def test_match_muzzle_frame_sets(synthetic_textured_images, tmp_path):
    """Verify 5x5 All-Pairs comparison between frame sets."""
    img1, img2 = synthetic_textured_images
    frames_a = [img1 for _ in range(5)]
    frames_b = [img2 for _ in range(5)]

    res = match_muzzle_frame_sets(frames_a, frames_b, output_dir=tmp_path)

    assert "best_pair" in res
    assert "median_score" in res
    assert "best_score" in res
    assert "composite_score" in res
    assert "inlier_count" in res
    assert "inlier_ratio" in res
    assert "matched_minutiae_fraction" in res
    assert "method_a" in res
    assert "method_b" in res
    assert "visual_url" in res
    assert "pair_scores" in res
    assert "is_match" in res
    assert "verdict" in res

    # 5x5 pair matrix dimensions
    assert len(res["pair_scores"]) == 5
    assert len(res["pair_scores"][0]) == 5

    # Check best pair structure
    bp = res["best_pair"]
    assert "frame_a_idx" in bp
    assert "frame_b_idx" in bp
    assert "score" in bp
    assert "score_method_a" in bp
    assert "score_method_b" in bp

    # Check visualization output created
    vis_files = list((tmp_path / "match_visualizations").glob("*.jpg"))
    assert len(vis_files) >= 1
    vis_img = cv2.imread(str(vis_files[0]))
    assert vis_img is not None
    # 448 + 56 banner height = 504
    assert vis_img.shape[0] == 504
    assert vis_img.shape[1] == 448 * 2


def test_load_matcher_config(tmp_path):
    """Verify matcher config loading with calibrated vs uncalibrated files."""
    # When missing
    cfg_none = load_matcher_config(tmp_path / "non_existent.json")
    assert cfg_none == {"calibrated": False, "status": "UNCALIBRATED", "threshold": None}

    # When present and calibrated
    cfg_file = tmp_path / "matcher_config.json"
    with open(cfg_file, "w") as f:
        json.dump({"calibrated": True, "status": "CALIBRATED", "threshold": 0.42}, f)

    cfg_loaded = load_matcher_config(cfg_file)
    assert cfg_loaded["calibrated"] is True
    assert cfg_loaded["threshold"] == 0.42


def test_server_matcher_config_endpoint():
    """Verify GET /api/video/matcher-config endpoint returns valid dict."""
    cfg = asyncio.run(get_matcher_config_endpoint())
    assert "calibrated" in cfg
    assert "status" in cfg
    assert "threshold" in cfg


def test_server_match_endpoint_with_frames(synthetic_textured_images):
    """Verify POST /api/video/match endpoint with pre-extracted frame data."""
    img1, img2 = synthetic_textured_images
    req = VideoMatchRequest(
        frames_a=[{"enhanced_gray": img1}],
        frames_b=[{"enhanced_gray": img2}],
    )
    res = asyncio.run(match_video_endpoint(req))
    assert "best_pair" in res
    assert "composite_score" in res
    assert "visual_url" in res
    assert "verdict" in res


def test_server_match_endpoint_validation():
    """Verify POST /api/video/match endpoint validates missing inputs."""
    # Empty request -> 400
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(match_video_endpoint(VideoMatchRequest()))
    assert exc_info.value.status_code == 400

    # Non-existent video path -> 404
    with pytest.raises(HTTPException) as exc_404:
        asyncio.run(match_video_endpoint(VideoMatchRequest(
            video_a_path="/tmp/non_existent_video_12345.mp4",
            video_b_path="/tmp/non_existent_video_67890.mp4",
        )))
    assert exc_404.value.status_code == 404
