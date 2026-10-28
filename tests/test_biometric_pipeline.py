"""
Comprehensive Unit and Integration Tests for Cattle Muzzle Biometric Pipeline
=============================================================================
Scope:
1. Skeleton pruning (groove_graph.prune_skeleton):
   - Spur removal (< 8px, configurable min_spur_length)
   - Small component removal (< 30px, configurable min_component_size)
   - Preservation of main trunks, long branches, and closed loop cycles
2. Canonical alignment (video_groove_extractor.align_muzzle_canonical):
   - Nostril landmark detection & horizontal leveling
   - Mask principal axis central moments fallback
   - Normalization to target size (448x448)
3. Biometric Matcher (muzzle_matcher.py):
   - Method A: SIFT/ORB + RANSAC keypoint matching (same vs different cattle)
   - Method B: Minutiae topological alignment (endpoints, junctions, orientations)
   - 5x5 All-Pairs frame comparison & visualization generation
4. API Endpoints (server.py):
   - GET /api/video/matcher-config
   - POST /api/video/match
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

# Ensure repository root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from groove_graph import (
    prune_skeleton,
    extract_skeleton_graph,
    compute_skeleton_neighbor_degrees,
)
from video_groove_extractor import align_muzzle_canonical
from muzzle_matcher import (
    match_sift_ransac,
    extract_minutiae_points,
    match_minutiae,
    match_muzzle_frame_sets,
    load_matcher_config,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_DUPLICATE_THRESHOLD,
)
from server import (
    get_matcher_config_endpoint,
    match_video_endpoint,
    VideoMatchRequest,
)


# =============================================================================
# GROUP A: SKELETON PRUNING TESTS
# =============================================================================

class TestSkeletonPruning:
    """Tests for prune_skeleton in groove_graph.py."""

    def test_prune_skeleton_removes_short_spurs_default(self):
        """
        Verify prune_skeleton removes spurs shorter than 8 px and preserves
        branches >= 8 px by default.
        """
        skel = np.zeros((100, 100), dtype=np.uint8)
        # Main horizontal trunk: y=50, from x=20 to x=80 (length = 61 px)
        cv2.line(skel, (20, 50), (80, 50), 255, 1)

        # Short spur: 4 pixels long, connected at (35, 50) going downward to (35, 54)
        cv2.line(skel, (35, 50), (35, 54), 255, 1)

        # Long branch: 15 pixels long, connected at (60, 50) going upward to (60, 35)
        cv2.line(skel, (60, 50), (60, 35), 255, 1)

        pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=20)

        # The short spur tip at (35, 54) must be removed
        assert pruned[54, 35] == 0, "Spur tip (length 4 < 8) should be pruned"
        assert pruned[52, 35] == 0, "Spur intermediate pixel should be pruned"

        # The long branch tip at (60, 35) must be preserved
        assert pruned[35, 60] == 255, "Long branch tip (length 15 >= 8) should be preserved"
        # Trunk center should be preserved
        assert pruned[50, 50] == 255, "Trunk pixel should be preserved"

    def test_prune_skeleton_custom_min_spur_length(self):
        """Verify custom min_spur_length parameter is honored."""
        skel = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(skel, (20, 50), (80, 50), 255, 1)  # Trunk length 61
        # Spur of length 6 px: from (40, 50) to (40, 56)
        cv2.line(skel, (40, 50), (40, 56), 255, 1)

        # With min_spur_length = 4 (spur length 6 >= 4), spur should be KEPT
        pruned_keep = prune_skeleton(skel, min_spur_length=4, min_component_size=10)
        assert pruned_keep[56, 40] == 255, "Spur of length 6 should be kept when min_spur_length=4"

        # With min_spur_length = 8 (spur length 6 < 8), spur should be REMOVED
        pruned_remove = prune_skeleton(skel, min_spur_length=8, min_component_size=10)
        assert pruned_remove[56, 40] == 0, "Spur of length 6 should be pruned when min_spur_length=8"

    def test_prune_skeleton_removes_small_components(self):
        """
        Verify prune_skeleton removes connected components smaller than 30 px,
        while retaining components >= 30 px.
        """
        skel = np.zeros((120, 120), dtype=np.uint8)
        # Large component: length 50 px from (20, 20) to (70, 20)
        cv2.line(skel, (20, 20), (70, 20), 255, 1)

        # Small component: length 12 px (< 30) from (90, 80) to (90, 91)
        cv2.line(skel, (90, 80), (90, 91), 255, 1)

        # Prune with default min_component_size = 30
        pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=30)

        # Large component must be preserved
        assert np.count_nonzero(pruned[19:22, 20:71]) > 40
        # Small component must be completely erased
        assert np.count_nonzero(pruned[79:93, 89:92]) == 0

    def test_prune_skeleton_custom_min_component_size(self):
        """Verify custom min_component_size parameter is strictly respected."""
        skel = np.zeros((100, 100), dtype=np.uint8)
        # Component of exactly 20 pixels
        cv2.line(skel, (10, 50), (29, 50), 255, 1)

        # With min_component_size=15, component of 20 pixels is preserved
        pruned_small_thresh = prune_skeleton(skel, min_spur_length=0, min_component_size=15)
        assert np.count_nonzero(pruned_small_thresh) >= 18

        # With min_component_size=25, component of 20 pixels is eliminated
        pruned_large_thresh = prune_skeleton(skel, min_spur_length=0, min_component_size=25)
        assert np.count_nonzero(pruned_large_thresh) == 0

    def test_prune_skeleton_preserves_main_trunk_and_loops(self):
        """
        Verify that main trunks, long branches, and closed loop cycles are preserved,
        while attached short spurs are removed.
        """
        skel = np.zeros((120, 120), dtype=np.uint8)
        # Closed rectangular loop: 30x30 pixels (perimeter ~ 120 pixels)
        cv2.rectangle(skel, (30, 30), (60, 60), 255, 1)

        # Short spur protruding from the loop edge: (45, 30) going to (45, 25) (5 px)
        cv2.line(skel, (45, 30), (45, 25), 255, 1)

        # Long external branch connected to corner (60, 60) extending to (85, 60) (25 px)
        cv2.line(skel, (60, 60), (85, 60), 255, 1)

        pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=30)
        graph_res = extract_skeleton_graph(pruned, prune=False)

        # Spur tip must be removed
        assert pruned[25, 45] == 0, "Spur off the loop should be pruned"

        # The loop cycle must be preserved
        assert graph_res["descriptors"]["cycle_count"] >= 1, "Loop cycle must remain intact"

        # The long branch tip must be preserved
        assert pruned[60, 85] == 255, "Long external branch must be preserved"

    def test_prune_skeleton_empty_and_edge_cases(self):
        """Verify prune_skeleton handles edge cases without exception."""
        # Empty array
        res_empty = prune_skeleton(np.zeros((50, 50), dtype=np.uint8))
        assert res_empty.shape == (50, 50)
        assert np.count_nonzero(res_empty) == 0

        # None input
        res_none = prune_skeleton(None)
        assert res_none is not None

        # 3D image input conversion
        skel_3d = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.line(skel_3d, (10, 30), (50, 30), (255, 255, 255), 1)
        res_3d = prune_skeleton(skel_3d, min_spur_length=4, min_component_size=10)
        assert res_3d.ndim == 2
        assert np.count_nonzero(res_3d) > 0


# =============================================================================
# GROUP B: CANONICAL ALIGNMENT TESTS
# =============================================================================

class TestCanonicalAlignment:
    """Tests for align_muzzle_canonical in video_groove_extractor.py."""

    def test_align_muzzle_canonical_with_nostrils(self):
        """
        Verify align_muzzle_canonical detects two oriented nostril landmarks
        in the upper 40%, rotates the muzzle to horizontal, and resizes to 448x448.
        """
        h, w = 300, 300
        canvas = np.full((h, w, 3), 210, dtype=np.uint8)
        mask = np.full((h, w), 255, dtype=np.uint8)

        # Place two dark nostril circles in the upper 40% (y < 120):
        # Left nostril at (100, 50), Right nostril at (200, 70)
        # dx = 100, dy = 20 -> expected angle = arctan2(20, 100) ≈ 11.31 degrees
        cv2.circle(canvas, (100, 50), 14, (25, 25, 25), -1)
        cv2.circle(canvas, (200, 70), 14, (25, 25, 25), -1)

        aligned_bgr, aligned_mask, angle_deg = align_muzzle_canonical(
            canvas, mask=mask, target_size=(448, 448)
        )

        assert aligned_bgr.shape == (448, 448, 3), "Aligned image must be target_size 448x448"
        assert aligned_mask.shape == (448, 448), "Aligned mask must be target_size 448x448"
        assert abs(angle_deg - 11.31) < 2.5, f"Expected angle ~11.31 deg, got {angle_deg}"

    def test_align_muzzle_canonical_moments_fallback(self):
        """
        Verify align_muzzle_canonical falls back to mask central moments principal
        axis when nostril landmarks are absent.
        """
        h, w = 300, 300
        canvas = np.full((h, w, 3), 180, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Tilted ellipse at angle 20 degrees
        cv2.ellipse(mask, (150, 150), (60, 120), 20, 0, 360, 255, -1)

        aligned_bgr, aligned_mask, angle_deg = align_muzzle_canonical(
            canvas, mask=mask, target_size=(448, 448)
        )

        assert aligned_bgr.shape == (448, 448, 3)
        assert aligned_mask.shape == (448, 448)
        # Moments angle should orient the major axis (near 20 degrees)
        assert abs(angle_deg - 20.0) < 3.0, f"Expected moment angle ~20.0 deg, got {angle_deg}"

    def test_align_muzzle_canonical_custom_target_size_and_empty(self):
        """Verify target_size parameter and graceful handling of empty inputs."""
        # Custom size (256, 256)
        img = np.full((200, 200, 3), 150, dtype=np.uint8)
        aligned, m_out, _ = align_muzzle_canonical(img, target_size=(256, 256))
        assert aligned.shape == (256, 256, 3)
        assert m_out is None

        # None / empty input
        aligned_empty, _, ang = align_muzzle_canonical(np.zeros((0, 0, 3), dtype=np.uint8))
        assert aligned_empty.shape == (448, 448, 3)
        assert ang == 0.0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def realistic_muzzle_pair():
    """
    Generates a textured synthetic muzzle pattern and a slightly shifted &
    rotated replica (same cattle), plus an inverted noise pattern (distinct cattle).
    """
    np.random.seed(123)
    base = np.full((448, 448), 130, dtype=np.uint8)
    # Add rich groove-like lines and pore beads
    for i in range(25, 420, 22):
        cv2.line(base, (i, 40), (i + 12, 400), 45, 2)
        cv2.circle(base, (i, i), 9, 215, -1)
        cv2.circle(base, (448 - i, i), 7, 35, -1)
    noise = np.random.normal(0, 12, (448, 448)).astype(np.int16)
    same_a = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Replica: Shifted by (dx=4, dy=-3) and rotated by 2.0 degrees
    M = cv2.getRotationMatrix2D((224, 224), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    same_b = cv2.warpAffine(same_a, M, (448, 448))

    # Distinct: Inverted and heavily smoothed pattern (no matching keypoints)
    distinct = 255 - cv2.GaussianBlur(same_a, (35, 35), 12)
    return same_a, same_b, distinct


# =============================================================================
# GROUP C: BIOMETRIC MATCHER TESTS
# =============================================================================

class TestBiometricMatcher:
    """Tests for SIFT/ORB + RANSAC and Minutiae matcher in muzzle_matcher.py."""


    def test_same_image_vs_shifted_rotated_produces_match(self, realistic_muzzle_pair):
        """
        Verify Method A (SIFT/RANSAC) and Method B (Minutiae) on same cattle with
        slight affine perturbation yields high inliers and is_match=True.
        """
        same_a, same_b, _ = realistic_muzzle_pair

        res_a = match_sift_ransac(same_a, same_b)
        assert res_a["inlier_count"] >= 30, f"Expected >=30 inliers, got {res_a['inlier_count']}"
        assert res_a["inlier_ratio"] >= 0.60, f"Expected inlier ratio >= 0.60, got {res_a['inlier_ratio']}"
        assert res_a["score_method_a"] >= 0.70

        # Run frame sets comparison
        comp = match_muzzle_frame_sets(
            frames_a=[{"enhanced_gray": same_a}],
            frames_b=[{"enhanced_gray": same_b}],
        )
        assert comp["composite_score"] >= 0.50, f"Expected composite >= 0.50, got {comp['composite_score']}"
        assert comp["is_match"] is True
        assert comp["inlier_count"] >= 30
        assert "visual_url" in comp and comp["visual_url"] is not None

    def test_completely_different_muzzles_produce_no_match(self, realistic_muzzle_pair):
        """
        Verify comparing distinct cattle muzzles produces low inliers, low score,
        and is_match=False.
        """
        same_a, _, distinct = realistic_muzzle_pair

        res_diff = match_sift_ransac(same_a, distinct)
        inlier_cnt = res_diff["inlier_count"]
        assert inlier_cnt < 15, f"Expected <15 inliers for distinct, got {inlier_cnt}"
        assert res_diff["score_method_a"] < 0.40

        comp_diff = match_muzzle_frame_sets(
            frames_a=[{"enhanced_gray": same_a}],
            frames_b=[{"enhanced_gray": distinct}],
        )
        assert comp_diff["composite_score"] < 0.45
        assert comp_diff["is_match"] is False
        assert comp_diff["verdict"] in ["NO MATCH (DISTINCT CATTLE)", "DIFFERENT_CATTLE"]

    def test_minutiae_matching_with_orientation(self):
        """Verify topological minutiae matching with position and local orientation."""
        # Set A: 5 points
        mins_a = [
            {"x": 100.0, "y": 100.0, "type": "endpoint", "orientation": 0.0},
            {"x": 150.0, "y": 120.0, "type": "junction", "orientation": 45.0},
            {"x": 200.0, "y": 150.0, "type": "endpoint", "orientation": 90.0},
            {"x": 250.0, "y": 200.0, "type": "junction", "orientation": -30.0},
            {"x": 300.0, "y": 250.0, "type": "endpoint", "orientation": -60.0},
        ]
        # Set B: Same points shifted by (+5, -3) with small orientation perturbation (2 deg)
        mins_b = [
            {"x": 105.0, "y": 97.0, "type": "endpoint", "orientation": 2.0},
            {"x": 155.0, "y": 117.0, "type": "junction", "orientation": 47.0},
            {"x": 205.0, "y": 147.0, "type": "endpoint", "orientation": 92.0},
            {"x": 255.0, "y": 197.0, "type": "junction", "orientation": -28.0},
            {"x": 305.0, "y": 247.0, "type": "endpoint", "orientation": -58.0},
        ]

        res = match_minutiae(mins_a, mins_b, spatial_threshold=15.0, angular_threshold=25.0)
        assert res["matched_count"] >= 4, f"Expected >= 4 matched minutiae, got {res['matched_count']}"
        assert res["matched_fraction"] >= 0.80

    def test_keypoint_match_visualization_file_created(self, realistic_muzzle_pair, tmp_path):
        """Verify match_muzzle_frame_sets writes keypoint visualization file to disk."""
        same_a, same_b, _ = realistic_muzzle_pair
        res = match_muzzle_frame_sets(
            frames_a=[{"enhanced_gray": same_a}],
            frames_b=[{"enhanced_gray": same_b}],
            output_dir=tmp_path,
        )
        assert "visual_url" in res
        vis_file = tmp_path / "match_visualizations" / Path(res["visual_url"]).name
        assert vis_file.exists(), f"Visualization image {vis_file} must exist on disk"
        loaded = cv2.imread(str(vis_file))
        assert loaded is not None and loaded.shape[0] > 100


# =============================================================================
# GROUP D: API ENDPOINTS TESTS
# =============================================================================

class TestServerApiEndpoints:
    """Tests for GET /api/video/matcher-config and POST /api/video/match in server.py."""

    def test_matcher_config_endpoint_returns_valid_structure(self):
        """Verify GET /api/video/matcher-config returns valid dictionary structure."""
        cfg = asyncio.run(get_matcher_config_endpoint())
        assert isinstance(cfg, dict)
        assert "calibrated" in cfg
        assert isinstance(cfg["calibrated"], bool)
        assert "threshold" in cfg
        if cfg["calibrated"]:
            assert cfg["threshold"] is not None
            assert "eer" in cfg
        else:
            assert cfg.get("status") == "UNCALIBRATED"

    def test_match_video_endpoint_with_frame_sets(self, realistic_muzzle_pair):
        """Verify POST /api/video/match handles pre-extracted frame data successfully."""
        same_a, same_b, _ = realistic_muzzle_pair
        req = VideoMatchRequest(
            frames_a=[{"enhanced_gray": same_a}],
            frames_b=[{"enhanced_gray": same_b}],
        )
        res = asyncio.run(match_video_endpoint(req))

        assert isinstance(res, dict)
        assert "composite_score" in res
        assert "best_score" in res
        assert "median_score" in res
        assert "inlier_count" in res
        assert "inlier_ratio" in res
        assert "matched_minutiae_fraction" in res
        assert "method_a" in res
        assert "method_b" in res
        assert "visual_url" in res
        assert "verdict" in res
        assert "is_match" in res
        assert res["is_match"] is True

    def test_match_video_endpoint_validation_errors(self):
        """Verify POST /api/video/match properly validates missing or malformed inputs."""
        # 1. Neither frames nor video paths provided -> 400 Bad Request
        with pytest.raises(HTTPException) as exc_400:
            asyncio.run(match_video_endpoint(VideoMatchRequest()))
        assert exc_400.value.status_code == 400

        # 2. Non-existent video files -> 404 Not Found
        with pytest.raises(HTTPException) as exc_404:
            asyncio.run(match_video_endpoint(VideoMatchRequest(
                video_a_path="/tmp/fake_non_existent_cow_a_12345.mp4",
                video_b_path="/tmp/fake_non_existent_cow_b_67890.mp4",
            )))
        assert exc_404.value.status_code == 404
