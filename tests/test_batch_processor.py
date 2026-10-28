#!/usr/bin/env python3
"""
Test Suite for Batch Processing and Quality Assessment Module
=============================================================
Tests cattle muzzle image quality metrics (sharpness, exposure, contrast, area,
completeness), near-duplicate detection, and batch processing integration.

Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

# Ensure parent and test dirs are on sys.path
_test_dir = Path(__file__).resolve().parent
_base_dir = _test_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from batch_processor import (
    assess_image_quality,
    detect_near_duplicates,
    process_image_batch,
    DEFAULT_QUALITY_THRESHOLDS,
)
from template_schema import SCHEMA_VERSION


def create_crisp_muzzle_crop(size: Tuple[int, int] = (224, 224), seed: int = 42) -> np.ndarray:
    """
    Generates a synthetic high-contrast, crisp cattle muzzle crop with distinctive
    groove line topologies parameterized by seed.
    """
    rng = np.random.RandomState(seed)
    h, w = size
    base_val = rng.randint(60, 90)
    img = np.full((h, w, 3), base_val, dtype=np.uint8)

    # Realistic skin grain noise
    noise = rng.normal(0, 6, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 30, 130).astype(np.uint8)

    # Seed-dependent groove line count, frequencies, and phases
    freq_y = rng.uniform(8.0, 25.0)
    freq_x = rng.uniform(8.0, 25.0)
    phase_y = rng.uniform(0, np.pi * 2)
    step_y = rng.randint(14, 24)
    step_x = rng.randint(18, 30)

    # Wavy horizontal grooves
    for y in range(30, h - 30, step_y):
        pts = [(x, int(y + 9 * np.sin(x / freq_y + phase_y))) for x in range(30, w - 30, 6)]
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k + 1], (18, 18, 18), 2)

    # Interconnecting vertical grooves
    for x in range(35, w - 35, step_x):
        pts = [(int(x + 7 * np.cos(y / freq_x + phase_y)), y) for y in range(30, h - 30, 6)]
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k + 1], (22, 22, 22), 1)

    return img


def get_real_or_synthetic_crop() -> np.ndarray:
    """Loads a real muzzle crop from output directory if available, else generates synthetic."""
    real_path = _base_dir / "output" / "grooves_muzzleVideo" / "01_native_muzzle_crop.jpg"
    if real_path.is_file():
        img = cv2.imread(str(real_path))
        if img is not None and img.size > 0:
            return img
    return create_crisp_muzzle_crop()


# =============================================================================
# Test 1: High quality crisp muzzle crop is classified as usable
# =============================================================================
def test_high_quality_crisp_muzzle_crop_is_usable():
    """
    Test 1: Verifies that a high quality crisp muzzle crop has high Laplacian
    variance, valid muzzle area, optimal exposure, and is classified as usable
    with no rejection reasons.
    """
    # Test on synthetic crisp crop
    crisp_synthetic = create_crisp_muzzle_crop(seed=42)
    quality = assess_image_quality(crisp_synthetic)

    assert quality["is_usable"] is True, f"Expected usable, got reasons: {quality['rejection_reasons']}"
    assert len(quality["rejection_reasons"]) == 0
    assert quality["sharpness"] >= DEFAULT_QUALITY_THRESHOLDS["min_sharpness"]
    assert quality["muzzle_area_pct"] >= DEFAULT_QUALITY_THRESHOLDS["min_area_pct"]
    assert quality["exposure_clipping_ratio"] < DEFAULT_QUALITY_THRESHOLDS["max_underexposed_ratio"]
    assert quality["overall_quality_score"] >= 50.0
    assert quality["is_truncated"] is False

    # Test on real native muzzle crop if present
    real_crop = get_real_or_synthetic_crop()
    real_quality = assess_image_quality(real_crop)
    assert real_quality["is_usable"] is True
    assert len(real_quality["rejection_reasons"]) == 0
    assert real_quality["sharpness"] >= 30.0
    assert real_quality["overall_quality_score"] >= 50.0


# =============================================================================
# Test 2: Severe motion blur image is rejected with reason 'severe_blur'
# =============================================================================
def test_severe_motion_blur_rejected_with_reason_severe_blur():
    """
    Test 2: Verifies that an image degraded by severe motion blur (low Laplacian variance)
    is rejected and contains 'severe_blur' in rejection_reasons.
    """
    base_img = create_crisp_muzzle_crop()

    # Apply severe horizontal motion blur kernel + Gaussian blur
    kernel_size = 25
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(base_img, -1, kernel)
    blurred = cv2.GaussianBlur(blurred, (21, 21), 7.0)

    quality = assess_image_quality(blurred)

    assert quality["is_usable"] is False
    assert "severe_blur" in quality["rejection_reasons"]
    assert quality["sharpness"] < DEFAULT_QUALITY_THRESHOLDS["min_sharpness"]
    assert quality["overall_quality_score"] < 50.0


# =============================================================================
# Test 3: Severely underexposed / dark image is rejected with reason 'underexposed'
# =============================================================================
def test_severely_underexposed_image_rejected_with_reason_underexposed():
    """
    Test 3: Verifies that a severely underexposed/dark image (intensity clipped near 0)
    is rejected and contains 'underexposed' in rejection_reasons.
    """
    base_img = create_crisp_muzzle_crop()

    # Darken severely so mean is < 10 and under-exposure clipping ratio is high (> 80%)
    dark_img = (base_img * 0.05).astype(np.uint8)

    quality = assess_image_quality(dark_img)

    assert quality["is_usable"] is False
    assert "underexposed" in quality["rejection_reasons"]
    assert (
        quality["underexposed_ratio"] > DEFAULT_QUALITY_THRESHOLDS["max_underexposed_ratio"]
        or quality["mean_intensity"] < DEFAULT_QUALITY_THRESHOLDS["min_mean_intensity"]
    )
    assert quality["overall_quality_score"] < 50.0


# =============================================================================
# Test 4: Near-duplicate detection correctly identifies identical/nearly identical images
# =============================================================================
def test_near_duplicate_detection_identifies_identical_and_nearly_identical():
    """
    Test 4: Verifies that near-duplicate detection correctly identifies pairs of
    identical and slightly perturbed images exceeding the duplicate threshold (0.92).
    """
    base_img = create_crisp_muzzle_crop(seed=101)

    # Identical copy
    img_identical = base_img.copy()

    # Nearly identical copy (slight sensor Gaussian noise)
    noise = np.random.normal(0, 1.5, base_img.shape)
    img_near = np.clip(base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Test pair of identical images
    dups_identical = detect_near_duplicates([base_img, img_identical], threshold=0.92)
    assert len(dups_identical) == 1
    i, j, sim = dups_identical[0]
    assert i == 0 and j == 1
    assert sim >= 0.99

    # Test batch containing both identical and nearly identical
    batch = [base_img, img_identical, img_near]
    dups_batch = detect_near_duplicates(batch, threshold=0.92)

    # Expected pairs: (0, 1), (0, 2), (1, 2)
    pair_indices = [(p[0], p[1]) for p in dups_batch]
    assert (0, 1) in pair_indices
    assert (0, 2) in pair_indices
    assert (1, 2) in pair_indices
    for _, _, sim in dups_batch:
        assert sim >= 0.92


# =============================================================================
# Test 5: Distinct images are not flagged as duplicates
# =============================================================================
def test_distinct_images_not_flagged_as_duplicates():
    """
    Test 5: Verifies that genuinely distinct cattle muzzle images are not flagged
    as duplicates.
    """
    # Create distinctly structured muzzle patterns with different seeds
    img_a = create_crisp_muzzle_crop(seed=100)
    img_b = create_crisp_muzzle_crop(seed=500)
    img_c = create_crisp_muzzle_crop(seed=900)
    img_d = create_crisp_muzzle_crop(seed=1300)

    dups = detect_near_duplicates([img_a, img_b], threshold=0.92)
    assert len(dups) == 0, f"Expected 0 duplicates for distinct images, got {dups}"

    dups_multi = detect_near_duplicates([img_a, img_b, img_c, img_d], threshold=0.92)
    assert len(dups_multi) == 0, f"Expected 0 duplicates for distinct multi-images, got {dups_multi}"


# =============================================================================
# Test 6: Batch with 5 images, 3 images, or 1 image executes cleanly
# =============================================================================
@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_batch_processing_clean_execution(batch_size: int):
    """
    Test 6: Verifies that batches of 1, 3, and 5 images execute cleanly end-to-end,
    generating structured records, 5-stage outputs, counts, and canonical template.
    """
    images = [create_crisp_muzzle_crop(seed=100 + i * 300) for i in range(batch_size)]

    res = process_image_batch(images)

    # Validate top-level response contract
    assert "records" in res
    assert "usable_count" in res
    assert "submitted_count" in res
    assert "duplicate_count" in res
    assert "independent_view_count" in res
    assert "rejected_count" in res
    assert "canonical_template" in res

    assert res["submitted_count"] == batch_size
    assert len(res["records"]) == batch_size
    assert res["usable_count"] == batch_size
    assert res["independent_view_count"] == batch_size

    # Validate each record structure and 5-stage outputs
    for i, record in enumerate(res["records"]):
        assert record["image_index"] == i
        assert record["is_usable"] is True
        assert record["is_duplicate"] is False
        assert len(record["rejection_reasons"]) == 0
        assert record["dimensions"] == (224, 224)

        # 5-stage pipeline outputs check
        assert "clean_mask" in record
        assert "masked_image" in record
        assert "enhanced_response" in record
        assert "groove_mask" in record
        assert "groove_skeleton" in record
        assert "groove_overlay" in record

        assert record["clean_mask"].shape == (224, 224)
        assert record["enhanced_response"].shape == (224, 224)
        assert record["groove_mask"].shape == (224, 224)
        assert record["groove_skeleton"].shape == (224, 224)
        assert record["groove_overlay"].shape == (224, 224, 3)

        # Metrics check
        assert record["groove_pixel_count"] > 0
        assert record["skeleton_pixel_count"] > 0
        assert record["groove_coverage_pct"] > 0.0

    # Validate Canonical Template structure
    template = res["canonical_template"]
    assert template["schema_version"] == SCHEMA_VERSION
    assert template["batch_summary"]["submitted_count"] == batch_size
    assert template["batch_summary"]["usable_count"] == batch_size
    assert len(template["per_image_features"]) == batch_size


# =============================================================================
# Test 7: Output paths and artifacts are correctly created when output_dir is passed
# =============================================================================
def test_output_artifacts_created_when_output_dir_passed():
    """
    Test 7: Verifies that when output_dir is provided, all 5-stage image artifacts,
    per-image saved paths, batch_summary.json, and canonical_template.json are created.
    """
    img1 = create_crisp_muzzle_crop(seed=101)
    img2 = create_crisp_muzzle_crop(seed=505)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "batch_run_output"

        res = process_image_batch([img1, img2], output_dir=str(out_path))

        assert out_path.is_dir()

        # Check per-image saved files
        for i, rec in enumerate(res["records"]):
            saved = rec["saved_paths"]
            assert len(saved) > 0

            # Ensure the 4 primary pipeline stages are saved to disk
            for stage_key in ["groove_response", "groove_mask", "groove_skeleton", "groove_overlay"]:
                assert stage_key in saved, f"Stage key {stage_key} missing in saved_paths"
                filepath = Path(saved[stage_key])
                assert filepath.is_file(), f"File {filepath} does not exist"
                assert filepath.stat().st_size > 0, f"File {filepath} is empty"

        # Check batch_summary.json
        summary_file = out_path / "batch_summary.json"
        assert summary_file.is_file()
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
        assert summary_data["submitted_count"] == 2
        assert summary_data["usable_count"] == 2
        assert len(summary_data["records"]) == 2

        # Check canonical_template.json
        template_file = out_path / "canonical_template.json"
        assert template_file.is_file()
        with open(template_file, "r") as f:
            template_data = json.load(f)
        assert template_data["schema_version"] == SCHEMA_VERSION
        assert template_data["batch_summary"]["usable_count"] == 2


# =============================================================================
# Test 8: Truncated crop (mask touching multiple borders) is rejected
# =============================================================================
def test_truncated_crop_rejected_with_reason_truncated_crop():
    """
    Verifies that a mask touching multiple borders (>= 2) is flagged as a truncated
    crop and rejected with 'truncated_crop'.
    """
    img = create_crisp_muzzle_crop()
    h, w = img.shape[:2]

    # Create a mask that touches both top and left borders
    trunc_mask = np.zeros((h, w), dtype=np.uint8)
    trunc_mask[0 : int(h * 0.7), 0 : int(w * 0.7)] = 255

    quality = assess_image_quality(img, mask=trunc_mask)

    assert quality["is_usable"] is False
    assert "truncated_crop" in quality["rejection_reasons"]
    assert quality["is_truncated"] is True
    assert quality["touched_border_count"] >= 2


# =============================================================================
# Test 9: Overexposed image is rejected with reason 'overexposed'
# =============================================================================
def test_overexposed_image_rejected_with_reason_overexposed():
    """
    Verifies that a washed-out / overexposed image is rejected with 'overexposed'.
    """
    base_img = create_crisp_muzzle_crop()
    # Wash out image so pixel intensities > 245
    overexposed_img = np.clip(base_img.astype(np.int16) + 200, 0, 255).astype(np.uint8)

    quality = assess_image_quality(overexposed_img)

    assert quality["is_usable"] is False
    assert "overexposed" in quality["rejection_reasons"]


# =============================================================================
# Test 10: Insufficient muzzle area is rejected with reason 'insufficient_area'
# =============================================================================
def test_insufficient_muzzle_area_rejected():
    """
    Verifies that a crop with insufficient muzzle coverage (< 15%) is rejected.
    """
    img = create_crisp_muzzle_crop()
    h, w = img.shape[:2]

    # Tiny mask covering less than 5%
    tiny_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(tiny_mask, (w // 2, h // 2), int(min(h, w) * 0.1), 255, -1)

    quality = assess_image_quality(img, mask=tiny_mask)

    assert quality["is_usable"] is False
    assert "insufficient_area" in quality["rejection_reasons"]
    assert quality["muzzle_area_pct"] < 15.0


# =============================================================================
# Test 11: Batch duplicate resolution decrements independent_view_count
# =============================================================================
def test_batch_duplicate_resolution_accounting():
    """
    Verifies that in a batch containing near-duplicates, usable_count reflects all
    quality-passing images, duplicate_count reflects duplicates, and
    independent_view_count equals usable_count minus duplicates.
    """
    img_a = create_crisp_muzzle_crop(seed=10)
    # img_a_dup is near-duplicate of img_a
    img_a_dup = np.clip(img_a.astype(np.float32) + np.random.normal(0, 1.0, img_a.shape), 0, 255).astype(np.uint8)
    # img_b is distinct
    img_b = create_crisp_muzzle_crop(seed=800)

    res = process_image_batch([img_a, img_a_dup, img_b])

    assert res["submitted_count"] == 3
    assert res["usable_count"] == 3
    assert res["duplicate_count"] == 1
    assert res["independent_view_count"] == 2

    assert res["records"][0]["is_duplicate"] is False
    assert res["records"][1]["is_duplicate"] is True
    assert res["records"][2]["is_duplicate"] is False
