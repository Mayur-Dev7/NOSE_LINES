"""
Test Suite for Biometric Reference Template Storage & Two-Template Matcher
==========================================================================
Strictly scoped to /home/adis_mayur/F_CODES/nose_lines.

Tests:
- Test 1: Save and load template retains 100% of data and valid schema.
- Test 2: Comparing a template with itself produces composite match score ~1.0 (verdict: MATCH).
- Test 3: Comparing two synthetically distinct templates produces low match score (verdict: DISTINCT).
- Test 4: Missing or empty consensus features are handled gracefully without crash.
- Test 5: List saved templates returns list of existing template summaries.
- Additional edge-case tests: schema invalidation, missing file, coordinate encodings, angle boundaries.
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Ensure nose_lines is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from template_schema import SCHEMA_VERSION, make_empty_template
from template_matcher import (
    save_reference_template,
    load_reference_template,
    list_saved_templates,
    compare_two_templates,
    compute_global_descriptor_similarity,
    compute_recurring_structure_similarity,
    _angle_difference_deg,
    _parse_recurring_branch,
    KEY_GLOBAL_FEATURES,
    FORENSIC_DISCLAIMER,
)


def _build_synthetic_template(
    template_id: str,
    skeleton_pixels: float = 2400.0,
    density: float = 0.22,
    endpoints: float = 75.0,
    junctions: float = 45.0,
    branch_len: float = 25.0,
    cycles: float = 12.0,
    euler: float = -8.0,
    curvature: float = 0.42,
    mad_scale: float = 0.05,
    num_branches: int = 6,
) -> Dict[str, Any]:
    """Helper creating a fully populated canonical reference template for testing."""
    tmpl = make_empty_template(template_id=template_id, submitted_count=5)
    tmpl["batch_summary"] = {
        "submitted_count": 5,
        "processed_count": 5,
        "usable_count": 4,
        "independent_view_count": 3,
        "rejected_count": 1,
    }
    tmpl["quality_summary"] = {
        "mean_quality_score": 85.5,
        "median_sharpness": 320.0,
        "usable_ratio": 0.80,
        "duplicate_count": 0,
    }
    tmpl["consensus_features"] = {
        "skeleton_pixel_count": {"median": skeleton_pixels, "mad": skeleton_pixels * mad_scale, "mean": skeleton_pixels, "count": 4},
        "skeleton_density": {"median": density, "mad": density * mad_scale, "mean": density, "count": 4},
        "endpoint_count": {"median": endpoints, "mad": endpoints * mad_scale, "mean": endpoints, "count": 4},
        "junction_count": {"median": junctions, "mad": junctions * mad_scale, "mean": junctions, "count": 4},
        "mean_branch_length": {"median": branch_len, "mad": branch_len * mad_scale, "mean": branch_len, "count": 4},
        "cycle_count": {"median": cycles, "mad": cycles * mad_scale, "mean": cycles, "count": 4},
        "euler_characteristic": {"median": euler, "mad": abs(euler) * mad_scale, "mean": euler, "count": 4},
        "curvature_mean": {"median": curvature, "mad": curvature * mad_scale, "mean": curvature, "count": 4},
    }

    # Generate synthetic recurring branches
    branches = []
    for k in range(num_branches):
        branches.append({
            "branch_id": f"br_{k}",
            "centroid": [round(0.2 + (0.1 * k), 3), round(0.3 + (0.08 * k), 3)],
            "orientation": round(float((25 * k) % 180), 1),
            "length": 0.05,
        })
    tmpl["recurring_graph_structures"] = branches

    tmpl["feature_stability"] = {
        "overall_stability_score": 0.92,
        "confidence_level": "high_confidence",
    }
    return tmpl


# ============================================================================
# Test 1: Save and load template retains 100% of data and valid schema
# ============================================================================

def test_save_and_load_retains_all_data(tmp_path):
    """Test 1: Verify save and load maintains 100% data integrity and schema version."""
    tmpl = _build_synthetic_template("cow_ref_001")
    
    saved_path = save_reference_template(tmpl, output_dir=tmp_path)
    assert os.path.exists(saved_path)
    assert Path(saved_path).name == "cow_ref_001.json"

    loaded = load_reference_template(saved_path)

    # 100% Data retention check
    assert loaded == tmpl
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["template_id"] == "cow_ref_001"
    assert loaded["batch_summary"]["usable_count"] == 4
    assert len(loaded["recurring_graph_structures"]) == 6
    assert loaded["consensus_features"]["junction_count"]["median"] == 45.0


def test_load_by_template_id(tmp_path):
    """Verify load_reference_template can resolve an ID within a specified templates directory."""
    tmpl = _build_synthetic_template("cow_ref_id_test")
    save_reference_template(tmpl, output_dir=tmp_path)

    loaded = load_reference_template("cow_ref_id_test", templates_dir=tmp_path)
    assert loaded["template_id"] == "cow_ref_id_test"


def test_load_invalid_schema_or_missing_file(tmp_path):
    """Verify schema version validation and missing file handling."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        load_reference_template("non_existent_id", templates_dir=tmp_path)

    # 2. Missing schema version
    bad_file = tmp_path / "no_schema.json"
    bad_file.write_text(json.dumps({"template_id": "bad"}))
    with pytest.raises(ValueError, match="missing 'schema_version'"):
        load_reference_template(bad_file)

    # 3. Incompatible schema version
    incompatible_file = tmp_path / "incompatible.json"
    incompatible_file.write_text(json.dumps({"template_id": "bad", "schema_version": "99.0"}))
    with pytest.raises(ValueError, match="Incompatible schema_version"):
        load_reference_template(incompatible_file)


# ============================================================================
# Test 2: Comparing a template with itself produces composite match score ~1.0
# ============================================================================

def test_compare_template_with_itself_matches():
    """Test 2: Comparing a template with itself produces composite score ~1.0 (verdict: MATCH)."""
    tmpl = _build_synthetic_template("cow_match_self", num_branches=8)
    
    res = compare_two_templates(tmpl, tmpl, decision_threshold=0.72)

    assert res["is_match"] is True
    assert res["verdict"] == "MATCH (Same Cattle Candidate)"
    assert np.isclose(res["composite_score"], 1.0, atol=1e-3)
    assert np.isclose(res["global_descriptor_similarity"], 1.0, atol=1e-3)
    assert np.isclose(res["recurring_structure_similarity"], 1.0, atol=1e-3)
    assert res["recurring_structure_breakdown"]["matched_count"] == 8
    assert res["disclaimer"] == FORENSIC_DISCLAIMER

    # All feature z-scores should be 0.0 and similarities 1.0
    for feat_name, stat in res["feature_breakdown"].items():
        assert stat["status"] == "evaluated"
        assert stat["z_score"] == 0.0
        assert np.isclose(stat["similarity"], 1.0, atol=1e-3)


# ============================================================================
# Test 3: Comparing synthetically distinct templates produces low score
# ============================================================================

def test_compare_synthetically_distinct_templates_distinct():
    """Test 3: Synthetically distinct templates (500 vs 5000 px, 10 vs 80 junctions) produce DISTINCT verdict."""
    tmpl_a = _build_synthetic_template(
        template_id="animal_alpha",
        skeleton_pixels=500.0,
        density=0.06,
        endpoints=18.0,
        junctions=10.0,
        branch_len=11.0,
        cycles=1.0,
        euler=8.0,
        curvature=0.15,
        mad_scale=0.05,
        num_branches=4,
    )
    # Put branches for A in top-left
    tmpl_a["recurring_graph_structures"] = [
        {"centroid": [0.1, 0.1], "orientation": 10.0},
        {"centroid": [0.15, 0.12], "orientation": 20.0},
        {"centroid": [0.2, 0.15], "orientation": 30.0},
    ]

    tmpl_b = _build_synthetic_template(
        template_id="animal_beta",
        skeleton_pixels=5000.0,
        density=0.45,
        endpoints=180.0,
        junctions=80.0,
        branch_len=45.0,
        cycles=25.0,
        euler=-35.0,
        curvature=0.88,
        mad_scale=0.05,
        num_branches=4,
    )
    # Put branches for B far away in bottom-right with orthogonal angles
    tmpl_b["recurring_graph_structures"] = [
        {"centroid": [0.8, 0.8], "orientation": 85.0},
        {"centroid": [0.85, 0.82], "orientation": 95.0},
        {"centroid": [0.9, 0.85], "orientation": 105.0},
    ]

    res = compare_two_templates(tmpl_a, tmpl_b, decision_threshold=0.72)

    assert res["is_match"] is False
    assert res["verdict"] == "DISTINCT (Different Animals)"
    assert res["composite_score"] < 0.20
    assert res["global_descriptor_similarity"] < 0.20
    assert res["recurring_structure_similarity"] == 0.0
    assert res["recurring_structure_breakdown"]["matched_count"] == 0


# ============================================================================
# Test 4: Missing or empty consensus features are handled gracefully
# ============================================================================

def test_empty_consensus_features_handled_gracefully():
    """Test 4a: Completely empty consensus features and structures don't crash and return DISTINCT."""
    empty_a = make_empty_template("empty_a")
    empty_b = make_empty_template("empty_b")

    res = compare_two_templates(empty_a, empty_b)

    assert res["is_match"] is False
    assert res["verdict"] == "DISTINCT (Different Animals)"
    assert res["composite_score"] == 0.0
    assert res["global_descriptor_similarity"] == 0.0
    assert res["recurring_structure_similarity"] == 0.0


def test_asymmetric_missing_features_handled_gracefully():
    """Test 4b: Template A has features, Template B has empty consensus dict."""
    tmpl_a = _build_synthetic_template("has_features")
    tmpl_b = make_empty_template("no_features")

    res = compare_two_templates(tmpl_a, tmpl_b)

    assert res["is_match"] is False
    assert res["verdict"] == "DISTINCT (Different Animals)"
    assert res["composite_score"] == 0.0
    # Every feature status in breakdown should be 'missing_one'
    for feat_name, stat in res["feature_breakdown"].items():
        assert stat["status"] == "missing_one"
        assert stat["similarity"] == 0.0


def test_scalar_and_malformed_features_handled():
    """Test 4c: Features formatted as raw numbers or malformed dicts are parsed safely."""
    tmpl_a = make_empty_template("scalar_a")
    tmpl_a["consensus_features"] = {
        "skeleton_pixel_count": 2000.0,  # Scalar float
        "junction_count": 50,           # Scalar int
        "endpoint_count": None,         # Explicit None
        "curvature_mean": "malformed",  # Malformed string
    }

    tmpl_b = make_empty_template("scalar_b")
    tmpl_b["consensus_features"] = {
        "skeleton_pixel_count": 2000.0,
        "junction_count": 50,
        "endpoint_count": {"median": 60.0},
        "curvature_mean": {"median": 0.45},
    }

    res = compare_two_templates(tmpl_a, tmpl_b)
    # The two valid scalar features match perfectly (s_i = 1.0)
    assert res["feature_breakdown"]["skeleton_pixel_count"]["similarity"] == 1.0
    assert res["feature_breakdown"]["junction_count"]["similarity"] == 1.0
    assert res["feature_breakdown"]["endpoint_count"]["status"] == "missing_one"


# ============================================================================
# Test 5: List saved templates returns list of existing template summaries
# ============================================================================

def test_list_saved_templates_summaries(tmp_path):
    """Test 5: list_saved_templates correctly enumerates stored template metadata."""
    # Ensure empty directory returns []
    empty_list = list_saved_templates(tmp_path)
    assert empty_list == []

    # Save 3 templates with distinct properties
    t1 = _build_synthetic_template("cow_t1")
    t1["batch_summary"]["submitted_count"] = 6
    t1["batch_summary"]["usable_count"] = 5
    t1["feature_stability"]["overall_stability_score"] = 0.88
    t1["feature_stability"]["confidence_level"] = "high"

    t2 = _build_synthetic_template("cow_t2")
    t2["batch_summary"]["submitted_count"] = 4
    t2["batch_summary"]["usable_count"] = 3
    t2["feature_stability"]["overall_stability_score"] = 0.74
    t2["feature_stability"]["confidence_level"] = "medium"

    t3 = _build_synthetic_template("cow_t3")
    t3["batch_summary"]["submitted_count"] = 8
    t3["batch_summary"]["usable_count"] = 7
    t3["feature_stability"]["overall_stability_score"] = 0.95
    t3["feature_stability"]["confidence_level"] = "high"

    save_reference_template(t1, output_dir=tmp_path)
    save_reference_template(t2, output_dir=tmp_path)
    save_reference_template(t3, output_dir=tmp_path)

    summaries = list_saved_templates(tmp_path)
    assert len(summaries) == 3

    id_map = {s["id"]: s for s in summaries}
    assert "cow_t1" in id_map
    assert "cow_t2" in id_map
    assert "cow_t3" in id_map

    s1 = id_map["cow_t1"]
    assert s1["submitted_count"] == 6
    assert s1["usable_count"] == 5
    assert s1["overall_stability"] == 0.88
    assert s1["confidence_level"] == "high"
    assert s1["file_path"].endswith("cow_t1.json")

    s3 = id_map["cow_t3"]
    assert s3["submitted_count"] == 8
    assert s3["usable_count"] == 7
    assert s3["overall_stability"] == 0.95


# ============================================================================
# Additional Edge Case Tests: Branch Parsing & Boundary Matches
# ============================================================================

def test_branch_parsing_formats():
    """Verify various recurring branch coordinate and angle formats."""
    # 1. Dict with centroid
    b1 = _parse_recurring_branch({"centroid": [0.25, 0.35], "orientation": 45.0})
    assert b1 == (0.25, 0.35, 45.0)

    # 2. Dict with start and end
    b2 = _parse_recurring_branch({"start": [0.1, 0.2], "end": [0.3, 0.4]})
    assert b2 is not None
    assert np.isclose(b2[0], 0.2)
    assert np.isclose(b2[1], 0.3)
    assert np.isclose(b2[2], 45.0)

    # 3. List/tuple
    b3 = _parse_recurring_branch([0.5, 0.6, 90.0])
    assert b3 == (0.5, 0.6, 90.0)

    # 4. None / malformed
    assert _parse_recurring_branch(None) is None
    assert _parse_recurring_branch("invalid") is None


def test_angle_difference_symmetry():
    """Verify angle difference calculation for circular and line symmetry."""
    # Same line modulo 180
    assert _angle_difference_deg(10.0, 190.0) == 0.0
    # Modulo 360 wrap-around
    assert _angle_difference_deg(5.0, 355.0) == 10.0
    # Right angle
    assert _angle_difference_deg(0.0, 90.0) == 90.0
    # Small difference
    assert _angle_difference_deg(20.0, 35.0) == 15.0


def test_branch_matching_threshold_boundaries():
    """Verify branch matching right at the boundary (dist=0.15, angle=35 deg)."""
    # Exactly within threshold
    b_a = [{"centroid": [0.3, 0.3], "orientation": 20.0}]
    b_b_match = [{"centroid": [0.3 + 0.15, 0.3], "orientation": 20.0 + 35.0}]  # dist=0.15, angle_diff=35.0
    sim_match, breakdown_match = compute_recurring_structure_similarity(b_a, b_b_match)
    assert breakdown_match["matched_count"] == 1
    assert sim_match == 1.0

    # Exceeding spatial distance (dist=0.16)
    b_b_far = [{"centroid": [0.3 + 0.16, 0.3], "orientation": 20.0}]
    sim_far, breakdown_far = compute_recurring_structure_similarity(b_a, b_b_far)
    assert breakdown_far["matched_count"] == 0
    assert sim_far == 0.0

    # Exceeding angle diff (diff=36 deg)
    b_b_angle = [{"centroid": [0.3, 0.3], "orientation": 20.0 + 36.0}]
    sim_angle, breakdown_angle = compute_recurring_structure_similarity(b_a, b_b_angle)
    assert breakdown_angle["matched_count"] == 0
    assert sim_angle == 0.0


def test_custom_decision_threshold():
    """Verify that custom decision_threshold alters the verdict accordingly."""
    tmpl = _build_synthetic_template("custom_thresh")
    # Slightly perturbed copy
    tmpl_perturbed = _build_synthetic_template(
        "custom_thresh_pert",
        skeleton_pixels=2500.0,  # +100
        mad_scale=0.10,
    )
    res_default = compare_two_templates(tmpl, tmpl_perturbed, decision_threshold=0.72)
    score = res_default["composite_score"]

    # Threshold set below score -> MATCH
    res_low = compare_two_templates(tmpl, tmpl_perturbed, decision_threshold=score - 0.05)
    assert res_low["is_match"] is True
    assert res_low["verdict"] == "MATCH (Same Cattle Candidate)"

    # Threshold set above score -> DISTINCT
    res_high = compare_two_templates(tmpl, tmpl_perturbed, decision_threshold=score + 0.05)
    assert res_high["is_match"] is False
    assert res_high["verdict"] == "DISTINCT (Different Animals)"
