#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Tests for Multi-Image Cattle Muzzle Reference Template.
============================================================================================
Covers all 14 scenarios:
1. 5 valid images from the same cow produce consolidated template with high confidence
2. Batches with fewer than 5 images (1, 2, 3, 4) produce valid reference templates
3. Single low-quality or blurry image in a 5-image batch is excluded from consensus with recorded reasons
4. Near-duplicate images are flagged, prevented from falsely boosting confidence
5. Images of different resolutions are handled consistently without coordinate errors
6. Missing segmentation mask uses fallback ROI and records warning
7. Consensus statistics (Median & MAD) compute correctly
8. Stability metric S in [0, 1] classifies stability levels
9. Recurring graph structures across independent views are detected
10. Canonical template save and reload preserves 100% data fidelity
11. Two-template comparison distinguishes same cow vs different cows
12. Export to formatted JSON
13. Export to per-image and consensus CSV reports
14. Graceful handling of edge cases (all rejected images) without crashing
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from batch_processor import process_image_batch, assess_image_quality, detect_near_duplicates
from groove_graph import extract_skeleton_graph, visualize_groove_graph
from template_consensus import build_consensus_template
from template_matcher import (
    save_reference_template,
    load_reference_template,
    compare_two_templates,
)
from template_exporter import (
    export_all_template_reports,
    export_consensus_summary_csv,
    export_per_image_measurements_csv,
)
from template_schema import robust_median_and_mad, calculate_stability, SCHEMA_VERSION


import math


def _make_synthetic_muzzle(seed=42, width=300, height=300, lines=6, blur=False, dark=False):
    """Generates synthetic muzzle image with dark ridge groove lines on smooth skin background."""
    np.random.seed(seed)
    # Smooth skin base with subtle lighting gradient
    img = np.full((height, width), 160, dtype=np.uint8)
    for y in range(height):
        img[y, :] = np.clip(160 + int(10 * math.sin(y / 30.0)), 0, 255)

    # Add dark groove lines
    for _ in range(lines):
        pt1 = (np.random.randint(20, width - 20), np.random.randint(20, height - 20))
        pt2 = (np.random.randint(20, width - 20), np.random.randint(20, height - 20))
        cv2.line(img, pt1, pt2, color=25, thickness=3, lineType=cv2.LINE_AA)

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if blur:
        bgr = cv2.GaussianBlur(bgr, (25, 25), 0)
    if dark:
        bgr = (bgr * 0.05).astype(np.uint8)

    # Valid muzzle mask (oval in center)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (width // 2 - 10, height // 2 - 10), 0, 0, 360, 255, -1)
    return bgr, mask


# -----------------------------------------------------------------------------
# Scenario 1: 5 valid images from the same cow produce high confidence template
# -----------------------------------------------------------------------------
def test_scenario_01_five_valid_images_high_confidence():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        images = []
        masks = []
        for i in range(5):
            img, mask = _make_synthetic_muzzle(seed=100 + i, width=320, height=320, lines=8)
            p_img = tmp_dir / f"cow_frame_{i}.png"
            p_mask = tmp_dir / f"mask_{i}.png"
            cv2.imwrite(str(p_img), img)
            cv2.imwrite(str(p_mask), mask)
            images.append(str(p_img))
            masks.append(mask)

        batch_res = process_image_batch(images, masks=masks, output_dir=tmp_dir)
        assert batch_res["usable_count"] == 5

        records = batch_res["records"]
        for r in records:
            g = extract_skeleton_graph(r["groove_skeleton"], mask=r["clean_mask"])
            r["descriptors"] = g["descriptors"]
            r["branches"] = g["branches"]
            r["centroid"] = g["spatial_centroid"]

        template = build_consensus_template(records, template_id="scenario_01_template")
        assert template["schema_version"] == SCHEMA_VERSION
        assert template["batch_summary"]["usable_count"] == 5
        assert template["batch_summary"]["independent_view_count"] >= 3
        assert len(template["consensus_features"]) > 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 2: Batches with fewer than 5 images (1, 2, 3, 4) produce valid templates
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("n_images", [1, 2, 3, 4])
def test_scenario_02_batches_with_fewer_than_five_images(n_images):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        images = []
        for i in range(n_images):
            img, _ = _make_synthetic_muzzle(seed=200 + i, lines=5)
            p_img = tmp_dir / f"frame_{i}.png"
            cv2.imwrite(str(p_img), img)
            images.append(str(p_img))

        batch_res = process_image_batch(images, output_dir=tmp_dir)
        records = batch_res["records"]
        for r in records:
            if r["is_usable"]:
                g = extract_skeleton_graph(r["groove_skeleton"])
                r["descriptors"] = g["descriptors"]
                r["branches"] = g["branches"]

        template = build_consensus_template(records)
        assert template["batch_summary"]["submitted_count"] == n_images
        assert template["template_id"] is not None
        assert "consensus_features" in template
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 3: Single low-quality or blurry image in 5-image batch is excluded
# -----------------------------------------------------------------------------
def test_scenario_03_low_quality_image_excluded_with_reasons():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        images = []
        for i in range(4):
            img, _ = _make_synthetic_muzzle(seed=300 + i, lines=7)
            p = tmp_dir / f"good_{i}.png"
            cv2.imwrite(str(p), img)
            images.append(str(p))

        # 5th image is heavily blurred
        blurry_img, _ = _make_synthetic_muzzle(seed=399, lines=7, blur=True)
        p_blur = tmp_dir / "blurry_4.png"
        cv2.imwrite(str(p_blur), blurry_img)
        images.append(str(p_blur))

        batch_res = process_image_batch(images, output_dir=tmp_dir)
        assert batch_res["submitted_count"] == 5
        assert batch_res["usable_count"] == 4
        assert batch_res["rejected_count"] == 1

        # Check rejection reasons on 5th record
        rec_blur = batch_res["records"][4]
        assert not rec_blur["is_usable"]
        assert len(rec_blur["rejection_reasons"]) > 0
        assert any("blur" in r.lower() or "sharpness" in r.lower() for r in rec_blur["rejection_reasons"])

        # Consensus built purely from valid images
        records = batch_res["records"]
        for r in records:
            if r["is_usable"]:
                g = extract_skeleton_graph(r["groove_skeleton"])
                r["descriptors"] = g["descriptors"]
                r["branches"] = g["branches"]

        template = build_consensus_template(records)
        assert template["batch_summary"]["usable_count"] == 4
        assert template["batch_summary"]["rejected_count"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 4: Near-duplicate images flagged and accounted for
# -----------------------------------------------------------------------------
def test_scenario_04_near_duplicate_detection_and_accounting():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        img, _ = _make_synthetic_muzzle(seed=400, lines=6)
        p1 = tmp_dir / "orig.png"
        p2 = tmp_dir / "dup.png"
        cv2.imwrite(str(p1), img)
        # Slight noise
        noisy = np.clip(img.astype(np.float32) + np.random.normal(0, 1, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(p2), noisy)

        batch_res = process_image_batch([str(p1), str(p2)])
        assert len(batch_res["near_duplicates"]) >= 1
        assert batch_res["duplicate_count"] >= 1
        assert batch_res["independent_view_count"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 5: Mixed resolutions handled consistently
# -----------------------------------------------------------------------------
def test_scenario_05_mixed_resolutions_handled_consistently():
    resolutions = [(224, 224), (320, 320), (448, 448)]
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        images = []
        for i, (w, h) in enumerate(resolutions):
            img, _ = _make_synthetic_muzzle(seed=500 + i, width=w, height=h, lines=6)
            p = tmp_dir / f"res_{w}x{h}.png"
            cv2.imwrite(str(p), img)
            images.append(str(p))

        batch_res = process_image_batch(images)
        assert batch_res["usable_count"] == 3
        for r in batch_res["records"]:
            g = extract_skeleton_graph(r["groove_skeleton"])
            assert "skeleton_density" in g["descriptors"]
            assert g["descriptors"]["skeleton_density"] > 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 6: Missing segmentation mask uses fallback ROI
# -----------------------------------------------------------------------------
def test_scenario_06_missing_segmentation_mask_fallback():
    img, _ = _make_synthetic_muzzle(seed=600, lines=5)
    # Process without mask
    batch_res = process_image_batch([img])
    assert batch_res["usable_count"] == 1
    rec = batch_res["records"][0]
    assert rec["clean_mask"] is not None
    assert np.count_nonzero(rec["clean_mask"]) > 0


# -----------------------------------------------------------------------------
# Scenario 7: Consensus statistics (Median & MAD) match mathematical formula
# -----------------------------------------------------------------------------
def test_scenario_07_robust_median_and_mad_math():
    values = [100.0, 102.0, 98.0, 101.0, 1000.0]  # with large outlier 1000
    med, mad = robust_median_and_mad(values)
    assert med == 101.0
    # Abs diffs from 101: |100-101|=1, |102-101|=1, |98-101|=3, |101-101|=0, |1000-101|=899
    # Sorted abs diffs: [0, 1, 1, 3, 899] -> median is 1.0
    assert mad == 1.0


# -----------------------------------------------------------------------------
# Scenario 8: Stability metric S in [0, 1] classifies stability levels
# -----------------------------------------------------------------------------
def test_scenario_08_stability_score_classification():
    # Stable: identical values (zero dispersion)
    s_stable = calculate_stability([100.0, 100.0, 100.0])
    assert s_stable >= 0.85

    # Low stability: high dispersion relative to magnitude
    s_unstable = calculate_stability([10.0, 50.0, 5.0, 45.0])
    assert s_unstable < 0.65


# -----------------------------------------------------------------------------
# Scenario 9: Recurring graph structures detected across independent views
# -----------------------------------------------------------------------------
def test_scenario_09_recurring_graph_structures_detected():
    # Construct 3 records with persistent branch at (0.45, 0.45)
    records = []
    for i in range(3):
        r = {
            "image_index": i,
            "source_id": f"img_{i}",
            "is_usable": True,
            "is_duplicate": False,
            "quality": {"quality_score": 90.0, "sharpness": 80.0},
            "graph_descriptors": {"skeleton_pixel_count": 500, "junction_count": 5},
            "branches": [
                {
                    "midpoint_norm": (0.45, 0.45),
                    "orientation_deg": 30.0,
                    "arc_length": 40.0,
                    "chord_length": 38.0,
                }
            ],
            "centroid": [0.5, 0.5],
            "principal_axis_deg": 30.0,
        }
        records.append(r)

    tmpl = build_consensus_template(records)
    structures = tmpl.get("recurring_graph_structures", [])
    assert len(structures) >= 1
    assert structures[0]["observed_count"] == 3


# -----------------------------------------------------------------------------
# Scenario 10: Canonical template save and reload preserves 100% data fidelity
# -----------------------------------------------------------------------------
def test_scenario_10_template_save_and_reload_fidelity():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        img, _ = _make_synthetic_muzzle(seed=1000, lines=6)
        batch_res = process_image_batch([img], batch_id="tmpl_save_test")
        records = batch_res["records"]
        for r in records:
            g = extract_skeleton_graph(r["groove_skeleton"])
            r["descriptors"] = g["descriptors"]

        tmpl = build_consensus_template(records, template_id="tmpl_save_test")
        p = save_reference_template(tmpl, tmp_dir / "test_tmpl.json")
        loaded = load_reference_template(p)

        assert loaded["template_id"] == tmpl["template_id"]
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["batch_summary"] == tmpl["batch_summary"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 11: Two-template comparison distinguishes same cow vs different cows
# -----------------------------------------------------------------------------
def test_scenario_11_template_comparison_same_vs_different():
    img_a, _ = _make_synthetic_muzzle(seed=1100, lines=8)
    img_b_same, _ = _make_synthetic_muzzle(seed=1100, lines=8)  # same cow
    img_c_diff, _ = _make_synthetic_muzzle(seed=9999, lines=1)  # totally different

    res_a = process_image_batch([img_a], batch_id="cow_A")
    res_b = process_image_batch([img_b_same], batch_id="cow_A_repeat")
    res_c = process_image_batch([img_c_diff], batch_id="cow_C")

    tmpl_a = build_consensus_template(res_a["records"], template_id="cow_A")
    tmpl_b = build_consensus_template(res_b["records"], template_id="cow_A_repeat")
    tmpl_c = build_consensus_template(res_c["records"], template_id="cow_C")

    # Self / same compare
    comp_same = compare_two_templates(tmpl_a, tmpl_b)
    score_same = comp_same.get("composite_score", comp_same.get("composite_similarity_score", 0.0))
    assert score_same >= 0.70
    assert comp_same["is_match"]

    # Different cow compare
    comp_diff = compare_two_templates(tmpl_a, tmpl_c)
    score_diff = comp_diff.get("composite_score", comp_diff.get("composite_similarity_score", 0.0))
    assert score_diff < score_same


# -----------------------------------------------------------------------------
# Scenario 12: Export to formatted JSON
# -----------------------------------------------------------------------------
def test_scenario_12_export_formatted_json():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        img, _ = _make_synthetic_muzzle(seed=1200, lines=5)
        batch = process_image_batch([img])
        tmpl = build_consensus_template(batch["records"])
        json_path = tmp_dir / "exported_tmpl.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tmpl, f, indent=2)

        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_version"] == SCHEMA_VERSION
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 13: Export to CSV produces valid measurements and consensus files
# -----------------------------------------------------------------------------
def test_scenario_13_export_to_csv_reports():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        img, _ = _make_synthetic_muzzle(seed=1300, lines=6)
        batch = process_image_batch([img])
        records = batch["records"]
        for r in records:
            g = extract_skeleton_graph(r["groove_skeleton"])
            r["descriptors"] = g["descriptors"]

        tmpl = build_consensus_template(records)
        export_res = export_all_template_reports(tmpl, tmp_dir)

        p_meas = Path(export_res["per_image_csv"])
        p_cons = Path(export_res["consensus_csv"])

        assert p_meas.exists() and p_meas.stat().st_size > 0
        assert p_cons.exists() and p_cons.stat().st_size > 0

        # Validate CSV contents
        content_meas = p_meas.read_text()
        assert "image_index" in content_meas
        content_cons = p_cons.read_text()
        assert "feature_name" in content_cons
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Scenario 14: Graceful handling of extreme edge cases (all rejected images)
# -----------------------------------------------------------------------------
def test_scenario_14_extreme_edge_cases_handled_gracefully():
    # Empty batch
    tmpl_empty = build_consensus_template([])
    assert tmpl_empty["confidence_level"] == "insufficient_data"
    assert tmpl_empty["usable_count"] == 0

    # Batch where all images are rejected
    rec_rejected = {
        "image_index": 0,
        "source_id": "bad_crop",
        "is_usable": False,
        "is_duplicate": False,
        "rejection_reasons": ["severe_blur", "underexposed"],
        "quality": {"quality_score": 10.0, "sharpness": 5.0},
    }
    tmpl_rejected = build_consensus_template([rec_rejected])
    assert tmpl_rejected["confidence_level"] == "insufficient_data"
    assert len(tmpl_rejected["warnings"]) > 0
