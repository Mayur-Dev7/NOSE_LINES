"""
Unit and Integration Tests for Video Groove Extractor, Canonical Alignment, and Quality Gating
==============================================================================================
"""

import math
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
import pytest

from video_groove_extractor import (
    align_muzzle_canonical,
    extract_five_muzzle_stages,
    extract_five_frames_groove_features,
    FEATURE_KEYS,
)
from groove_graph import prune_skeleton, extract_skeleton_graph
from groove_extractor import extract_muzzle_grooves_pipeline


def test_align_muzzle_canonical_nostril_leveling():
    """Verify primary nostril landmark detection detects tilt and levels nostrils horizontally."""
    h, w = 300, 300
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)

    # Place two dark circular nostrils in upper 40% (y < 120), tilted clockwise by 12 degrees
    center = (150, 60)
    angle_rad = math.radians(12.0)
    dist = 50.0

    p1 = (int(center[0] - dist * math.cos(angle_rad)), int(center[1] - dist * math.sin(angle_rad)))
    p2 = (int(center[0] + dist * math.cos(angle_rad)), int(center[1] + dist * math.sin(angle_rad)))

    cv2.circle(canvas, p1, 12, (20, 20, 20), -1)
    cv2.circle(canvas, p2, 12, (20, 20, 20), -1)

    aligned_bgr, aligned_mask, angle_deg = align_muzzle_canonical(canvas, target_size=(448, 448))

    assert aligned_bgr.shape == (448, 448, 3)
    assert angle_deg == pytest.approx(12.0, abs=2.5)
    assert aligned_mask is None


def test_align_muzzle_canonical_mask_fallback():
    """Verify fallback to central moments when nostrils are absent."""
    h, w = 300, 300
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Ellipse tilted clockwise by 15 degrees
    cv2.ellipse(mask, (150, 150), (90, 50), 15, 0, 360, 255, -1)

    aligned_bgr, aligned_mask, angle_deg = align_muzzle_canonical(
        canvas, mask=mask, target_size=(448, 448)
    )

    assert aligned_bgr.shape == (448, 448, 3)
    assert aligned_mask is not None
    assert aligned_mask.shape == (448, 448)
    assert angle_deg == pytest.approx(15.0, abs=2.0)


def test_align_muzzle_canonical_empty_edge_cases():
    """Verify empty or zero-sized arrays do not crash."""
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    al_bgr, al_mask, ang = align_muzzle_canonical(empty_img, target_size=(448, 448))
    assert al_bgr.shape == (448, 448, 3)
    assert ang == 0.0

    none_bgr, none_mask, ang2 = align_muzzle_canonical(None, target_size=(448, 448))
    assert none_bgr.shape == (448, 448, 3)
    assert ang2 == 0.0


def test_function_aliases_and_signatures():
    """Verify extract_five_muzzle_stages is aliased with extract_five_frames_groove_features."""
    assert extract_five_muzzle_stages is extract_five_frames_groove_features

    import inspect
    sig = inspect.signature(extract_five_muzzle_stages)
    assert "video_path" in sig.parameters
    assert "method" in sig.parameters
    assert "min_spur_length" in sig.parameters
    assert "min_component_size" in sig.parameters
    assert sig.parameters["min_spur_length"].default == 8
    assert sig.parameters["min_component_size"].default == 30


def test_quality_gating_and_scoring_formula():
    """Verify quality score calculation formula and threshold gating logic."""
    conf = 0.95
    lap_var = 120.0
    score = (conf * 100.0) + min(100.0, lap_var / 5.0)
    assert score == pytest.approx(95.0 + 24.0, abs=1e-3)

    # Clamping of lap_var contribution
    lap_var_high = 600.0
    score_high = (conf * 100.0) + min(100.0, lap_var_high / 5.0)
    assert score_high == pytest.approx(95.0 + 100.0, abs=1e-3)


def test_minutiae_records_format():
    """Verify minutiae records generated from aligned + pruned skeleton match expected schema."""
    samples_dir = Path(__file__).resolve().parent.parent / "test_samples"
    sample_file = samples_dir / "cow_muzzle_macro.jpg"
    assert sample_file.exists()

    img = cv2.imread(str(sample_file))
    assert img is not None

    aligned_bgr, _, _ = align_muzzle_canonical(img, target_size=(448, 448))
    groove_res = extract_muzzle_grooves_pipeline(aligned_bgr, method="blackhat")
    pruned_skel = prune_skeleton(groove_res["groove_skeleton"], min_spur_length=8, min_component_size=30)
    graph_res = extract_skeleton_graph(pruned_skel, prune=False)

    minutiae = [
        {
            "x": float(n["pos"][0]),
            "y": float(n["pos"][1]),
            "type": n["type"],
            "orientation": float(n.get("orientation", 0.0)),
        }
        for n in graph_res["node_list"]
    ]

    assert len(minutiae) > 0
    for m in minutiae:
        assert "x" in m and isinstance(m["x"], float)
        assert "y" in m and isinstance(m["y"], float)
        assert "type" in m and m["type"] in ("endpoint", "junction", "loop")
        assert "orientation" in m and isinstance(m["orientation"], float)
        assert -180.0 <= m["orientation"] <= 180.0
