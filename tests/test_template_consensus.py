"""
Unit and Integration Tests for Biometric Consensus and Alignment Module
========================================================================
Validates multi-view consensus aggregation, spatial registration assessment,
branch feature correspondence, robust median/MAD statistics, and canonical
schema conformity.
Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Ensure nose_lines project root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pytest

from template_schema import SCHEMA_VERSION, make_empty_template
from template_consensus import (
    build_consensus_template,
    _compute_axial_median,
    _normalize_branch,
    _branches_match,
    _assess_spatial_registration,
    _mark_duplicates,
)
from template_exporter import export_all_template_reports


def _create_synthetic_record(
    index: int,
    is_usable: bool = True,
    is_duplicate: bool = False,
    quality_score: float = 0.90,
    sharpness: float = 85.0,
    skeleton_pixels: int = 1200,
    endpoints: int = 14,
    junctions: int = 8,
    components: int = 3,
    cycles: int = 2,
    mean_branch_len: float = 45.0,
    euler_char: int = -3,
    curvature: float = 0.05,
    branches: List[Dict[str, Any]] = None,
    centroid: List[float] = None,
    orientation_deg: float = 45.0,
    rejection_reasons: List[str] = None,
) -> Dict[str, Any]:
    """Helper to generate a mock per-image record from upstream pipeline stages."""
    if branches is None:
        branches = []
    if centroid is None:
        centroid = [0.50, 0.50]
    if rejection_reasons is None:
        rejection_reasons = []

    return {
        "image_index": index,
        "source_id": f"muzzle_capture_{index:03d}.jpg",
        "is_usable": is_usable,
        "is_duplicate": is_duplicate,
        "quality": {
            "quality_score": quality_score,
            "sharpness": sharpness,
            "contrast": 60.0,
            "valid_area_pct": 98.0,
        },
        "graph_descriptors": {
            "skeleton_pixel_count": skeleton_pixels,
            "skeleton_density": 0.15,
            "endpoint_count": endpoints,
            "junction_count": junctions,
            "connected_component_count": components,
            "cycle_count": cycles,
            "mean_branch_length": mean_branch_len,
            "euler_characteristic": euler_char,
            "curvature_mean": curvature,
        },
        "centroid_norm": centroid,
        "orientation_deg": orientation_deg,
        "branches": branches,
        "rejection_reasons": rejection_reasons,
    }


# -----------------------------------------------------------------------------
# Test 1: Batch of 5 consistent images produces high-confidence template
# -----------------------------------------------------------------------------
def test_consistent_batch_high_confidence():
    """5 consistent usable views yield high confidence, stable medians, and low MAD."""
    records = []
    np.random.seed(42)
    base_pixels = 1200
    for i in range(5):
        # Introduce tiny realistic measurement noise (+/- 4 pixels)
        noise = int(np.random.randint(-4, 5))
        rec = _create_synthetic_record(
            index=i,
            is_usable=True,
            is_duplicate=False,
            quality_score=0.92,
            sharpness=86.0,
            skeleton_pixels=base_pixels + noise,
            endpoints=14,
            junctions=8,
            centroid=[0.50 + i * 0.005, 0.50 - i * 0.005],
            orientation_deg=45.0 + (i - 2) * 1.0,
        )
        records.append(rec)

    template = build_consensus_template(records, template_id="test_tmpl_01")

    assert template["template_id"] == "test_tmpl_01"
    assert template["batch_summary"]["submitted_count"] == 5
    assert template["batch_summary"]["usable_count"] == 5
    assert template["batch_summary"]["independent_view_count"] == 5
    assert template["batch_summary"]["rejected_count"] == 0

    assert template["quality_summary"]["usable_ratio"] == 1.0
    assert template["quality_summary"]["mean_quality_score"] == 0.92

    # Consensus features checks
    skel_stat = template["consensus_features"]["skeleton_pixel_count"]
    assert 1195.0 <= skel_stat["median"] <= 1205.0
    assert skel_stat["mad"] <= 5.0
    assert skel_stat["stability"] >= 0.90
    assert skel_stat["count"] == 5

    # Overall stability and confidence level
    stability = template["feature_stability"]
    assert stability["overall_stability_score"] >= 0.85
    assert stability["confidence_level"] == "high"

    # Spatial alignment assessment
    align = template["alignment_summary"]
    assert align["registration_attempted"] is True
    assert align["reliable_alignment"] is True
    assert align["quality_metric"] >= 0.70


# -----------------------------------------------------------------------------
# Test 2: Low-quality image excluded from robust consensus
# -----------------------------------------------------------------------------
def test_low_quality_image_excluded():
    """1 degraded image among 4 valid ones is excluded from consensus medians."""
    records = []
    for i in range(4):
        rec = _create_synthetic_record(
            index=i,
            is_usable=True,
            is_duplicate=False,
            quality_score=0.88,
            sharpness=80.0,
            skeleton_pixels=1000,
            endpoints=12,
            junctions=6,
        )
        records.append(rec)

    # 5th image is blurred / unusable with an extreme outlier measurement
    corrupt_rec = _create_synthetic_record(
        index=4,
        is_usable=False,
        quality_score=0.15,
        sharpness=12.0,
        skeleton_pixels=30,  # Extreme outlier that would skew naive average
        endpoints=1,
        junctions=0,
        rejection_reasons=["severe_motion_blur", "insufficient_sharpness"],
    )
    records.append(corrupt_rec)

    template = build_consensus_template(records)

    assert template["batch_summary"]["submitted_count"] == 5
    assert template["batch_summary"]["usable_count"] == 4
    assert template["batch_summary"]["rejected_count"] == 1
    assert template["batch_summary"]["independent_view_count"] == 4

    # The corrupted image must not corrupt the consensus stats
    skel_stat = template["consensus_features"]["skeleton_pixel_count"]
    assert skel_stat["median"] == 1000.0
    assert skel_stat["count"] == 4
    assert skel_stat["mad"] == 0.0

    # Source images metadata preserves rejection info
    rejected_src = next(s for s in template["source_images"] if s["image_index"] == 4)
    assert rejected_src["is_usable"] is False
    assert "severe_motion_blur" in rejected_src["rejection_reasons"]

    # Template retains high confidence because 4/5 usable views remain
    assert template["feature_stability"]["confidence_level"] == "high"


# -----------------------------------------------------------------------------
# Test 3: Near-duplicates correctly reduce independent view count
# -----------------------------------------------------------------------------
def test_near_duplicates_reported():
    """Near-duplicate burst frames reduce independent_view_count below submitted_count."""
    records = [
        _create_synthetic_record(0, is_usable=True, is_duplicate=False, skeleton_pixels=1100),
        _create_synthetic_record(1, is_usable=True, is_duplicate=False, skeleton_pixels=1105),
        _create_synthetic_record(2, is_usable=True, is_duplicate=False, skeleton_pixels=1095),
        # Explicit duplicate flag
        _create_synthetic_record(3, is_usable=True, is_duplicate=True, skeleton_pixels=1100),
        # Identical near-duplicate of image 0 (matching source_id)
        _create_synthetic_record(4, is_usable=True, is_duplicate=False, skeleton_pixels=1100),
    ]
    # Set identical source_id to trigger duplicate detection on image 4
    records[4]["source_id"] = records[0]["source_id"]

    template = build_consensus_template(records)

    assert template["batch_summary"]["submitted_count"] == 5
    assert template["batch_summary"]["usable_count"] == 5
    assert template["batch_summary"]["independent_view_count"] == 3
    assert template["batch_summary"]["independent_view_count"] < template["batch_summary"]["submitted_count"]
    assert template["quality_summary"]["duplicate_count"] == 2


# -----------------------------------------------------------------------------
# Test 4: Batch with only 1 usable image sets low confidence & warning
# -----------------------------------------------------------------------------
def test_single_usable_image_low_confidence_and_warning():
    """Batch with only 1 usable image sets confidence_level='low' and adds evidence warning."""
    records = [
        _create_synthetic_record(0, is_usable=True, skeleton_pixels=1150),
        _create_synthetic_record(1, is_usable=False, rejection_reasons=["underexposed"]),
        _create_synthetic_record(2, is_usable=False, rejection_reasons=["specular_glare"]),
    ]

    template = build_consensus_template(records)

    assert template["batch_summary"]["usable_count"] == 1
    assert template["batch_summary"]["independent_view_count"] == 1
    assert template["feature_stability"]["confidence_level"] == "low"

    # Must contain the exact required warning string
    expected_warning = "Template has limited multi-image evidence (only 1 independent usable view)."
    assert any(expected_warning in w for w in template["warnings"])

    # Consensus features are computed for the single observation
    assert template["consensus_features"]["skeleton_pixel_count"]["count"] == 1
    assert template["consensus_features"]["skeleton_pixel_count"]["median"] == 1150.0
    assert template["consensus_features"]["skeleton_pixel_count"]["mad"] == 0.0


# -----------------------------------------------------------------------------
# Test 5: Batch with 0 usable images safely returns empty consensus
# -----------------------------------------------------------------------------
def test_zero_usable_images_safe_empty_template():
    """Batch with 0 usable images returns empty consensus and does not raise exception."""
    records = [
        _create_synthetic_record(0, is_usable=False, rejection_reasons=["crop_failed"]),
        _create_synthetic_record(1, is_usable=False, rejection_reasons=["severe_blur"]),
    ]

    template = build_consensus_template(records)

    assert template["batch_summary"]["submitted_count"] == 2
    assert template["batch_summary"]["usable_count"] == 0
    assert template["batch_summary"]["independent_view_count"] == 0
    assert template["batch_summary"]["rejected_count"] == 2

    assert template["feature_stability"]["confidence_level"] == "insufficient_data"
    assert template["feature_stability"]["overall_stability_score"] == 0.0
    assert template["consensus_features"] == {}
    assert template["recurring_graph_structures"] == []
    assert len(template["warnings"]) > 0

    # Also test empty list input
    empty_template = build_consensus_template([])
    assert empty_template["batch_summary"]["submitted_count"] == 0
    assert empty_template["batch_summary"]["usable_count"] == 0
    assert empty_template["feature_stability"]["confidence_level"] == "insufficient_data"


# -----------------------------------------------------------------------------
# Test 6: Recurring structures identified across images
# -----------------------------------------------------------------------------
def test_recurring_structures_identified():
    """Recurring branch structures observed across multiple images are correctly matched and aggregated."""
    # Define three recurring branch lines:
    # Line 1: Present in all 3 images (k=3, recurrence=1.0)
    # Line 2: Present in 2 images (k=2, recurrence=0.667)
    # Line 3: Noise line in only 1 image (k=1, recurrence=0.333) -> filtered out (< 0.5)

    branches_img0 = [
        {"midpoint": [0.35, 0.45], "orientation_deg": 25.0, "length": 0.30},
        {"midpoint": [0.65, 0.75], "orientation_deg": 115.0, "length": 0.20},
    ]
    branches_img1 = [
        {"midpoint": [0.36, 0.46], "orientation_deg": 27.0, "length": 0.31},
        {"midpoint": [0.66, 0.74], "orientation_deg": 112.0, "length": 0.19},
    ]
    branches_img2 = [
        {"midpoint": [0.34, 0.44], "orientation_deg": 24.0, "length": 0.29},
        # Noise line unique to image 2
        {"midpoint": [0.10, 0.85], "orientation_deg": 80.0, "length": 0.12},
    ]

    records = [
        _create_synthetic_record(0, is_usable=True, branches=branches_img0),
        _create_synthetic_record(1, is_usable=True, branches=branches_img1),
        _create_synthetic_record(2, is_usable=True, branches=branches_img2),
    ]

    template = build_consensus_template(records, min_recurrence_ratio=0.5)
    structures = template["recurring_graph_structures"]

    # Only lines 1 and 2 should recur with recurrence ratio >= 0.5
    assert len(structures) == 2

    # Top recurring structure should be Line 1 (observed in all 3 views)
    s1 = structures[0]
    assert s1["structure_id"] == "recurring_ridge_01"
    assert s1["observed_count"] == 3
    assert s1["recurrence_ratio"] == 1.0
    assert np.isclose(s1["centroid_norm"][0], 0.35, atol=0.03)
    assert np.isclose(s1["centroid_norm"][1], 0.45, atol=0.03)
    assert np.isclose(s1["median_length"], 0.30, atol=0.03)
    assert np.isclose(s1["median_orientation_deg"], 25.0, atol=3.0)

    # Second recurring structure should be Line 2 (observed in 2 views)
    s2 = structures[1]
    assert s2["structure_id"] == "recurring_ridge_02"
    assert s2["observed_count"] == 2
    assert s2["recurrence_ratio"] == round(2 / 3, 3)
    assert np.isclose(s2["centroid_norm"][0], 0.655, atol=0.03)
    assert np.isclose(s2["centroid_norm"][1], 0.745, atol=0.03)


# -----------------------------------------------------------------------------
# Test 7: Canonical schema fields are all present and valid
# -----------------------------------------------------------------------------
def test_canonical_schema_conformity():
    """Validates complete presence of canonical keys and JSON serialization."""
    records = [_create_synthetic_record(i) for i in range(3)]
    template = build_consensus_template(records, template_id="canonical_test_tmpl")

    required_keys = [
        "schema_version",
        "template_id",
        "created_at",
        "source_images",
        "batch_summary",
        "quality_summary",
        "per_image_features",
        "alignment_summary",
        "consensus_features",
        "recurring_graph_structures",
        "feature_stability",
        "processing_config",
        "warnings",
    ]
    for key in required_keys:
        assert key in template, f"Missing required canonical top-level key: {key}"

    assert template["schema_version"] == SCHEMA_VERSION
    assert isinstance(template["source_images"], list)
    assert isinstance(template["per_image_features"], list)
    assert isinstance(template["consensus_features"], dict)
    assert isinstance(template["recurring_graph_structures"], list)
    assert isinstance(template["warnings"], list)

    # Validate JSON serializability
    serialized = json.dumps(template)
    deserialized = json.loads(serialized)
    assert deserialized["template_id"] == "canonical_test_tmpl"


# -----------------------------------------------------------------------------
# Test 8: End-to-end integration with template_exporter.py
# -----------------------------------------------------------------------------
def test_export_all_template_reports_integration():
    """Validates full compatibility with template_exporter (JSON, per-image CSV, consensus CSV)."""
    records = [_create_synthetic_record(i) for i in range(4)]
    template = build_consensus_template(records, template_id="export_test_tmpl")

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_all_template_reports(template, export_dir=tmp_dir)

        json_path = Path(paths["json"])
        per_img_csv = Path(paths["per_image_csv"])
        consensus_csv = Path(paths["consensus_csv"])

        assert json_path.exists() and json_path.stat().st_size > 100
        assert per_img_csv.exists() and per_img_csv.stat().st_size > 100
        assert consensus_csv.exists() and consensus_csv.stat().st_size > 100

        # Read back json to ensure validity
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["template_id"] == "export_test_tmpl"


# -----------------------------------------------------------------------------
# Test 9: Spatial alignment assessment flags misaligned views
# -----------------------------------------------------------------------------
def test_misaligned_batch_flags_unreliable():
    """Centroid offsets or principal axis divergences >= tolerance result in reliable_alignment=False."""
    records = [
        _create_synthetic_record(0, centroid=[0.20, 0.20], orientation_deg=10.0),
        _create_synthetic_record(1, centroid=[0.80, 0.80], orientation_deg=80.0),
    ]

    template = build_consensus_template(records)
    align = template["alignment_summary"]

    assert align["registration_attempted"] is True
    assert align["reliable_alignment"] is False
    assert align["quality_metric"] < 0.70
    assert "Registration uncertain" in align["reason"]
    assert any("Spatial registration across usable images is unreliable" in w for w in template["warnings"])


# -----------------------------------------------------------------------------
# Test 10: High-dispersion noisy batch yields low confidence
# -----------------------------------------------------------------------------
def test_high_dispersion_yields_low_confidence():
    """Widely varying descriptor measurements across views lower stability and trigger low confidence."""
    # 4 images with wildly fluctuating descriptors across views
    records = []
    fluctuating_descriptors = [
        {
            "skeleton_pixel_count": 200,
            "skeleton_density": 0.02,
            "endpoint_count": 2,
            "junction_count": 1,
            "connected_component_count": 1,
            "cycle_count": 0,
            "mean_branch_length": 8.0,
            "euler_characteristic": 1,
            "curvature_mean": 0.01,
        },
        {
            "skeleton_pixel_count": 2500,
            "skeleton_density": 0.25,
            "endpoint_count": 45,
            "junction_count": 28,
            "connected_component_count": 12,
            "cycle_count": 7,
            "mean_branch_length": 75.0,
            "euler_characteristic": -15,
            "curvature_mean": 0.12,
        },
        {
            "skeleton_pixel_count": 8000,
            "skeleton_density": 0.65,
            "endpoint_count": 130,
            "junction_count": 85,
            "connected_component_count": 35,
            "cycle_count": 24,
            "mean_branch_length": 140.0,
            "euler_characteristic": -50,
            "curvature_mean": 0.35,
        },
        {
            "skeleton_pixel_count": 450,
            "skeleton_density": 0.05,
            "endpoint_count": 6,
            "junction_count": 3,
            "connected_component_count": 2,
            "cycle_count": 1,
            "mean_branch_length": 12.0,
            "euler_characteristic": 2,
            "curvature_mean": 0.02,
        },
    ]
    for i, desc in enumerate(fluctuating_descriptors):
        rec = {
            "image_index": i,
            "source_id": f"noisy_{i:03d}.jpg",
            "is_usable": True,
            "is_duplicate": False,
            "quality": {"quality_score": 0.70, "sharpness": 60.0},
            "graph_descriptors": desc,
            "centroid_norm": [0.5, 0.5],
            "orientation_deg": 45.0,
        }
        records.append(rec)

    template = build_consensus_template(records)
    stability = template["feature_stability"]

    assert stability["overall_stability_score"] < 0.65
    assert stability["confidence_level"] == "low"
    assert any("Overall topological stability is low" in w for w in template["warnings"])
