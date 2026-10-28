"""
Unit and Integration Tests for Cattle Muzzle Groove Extraction Engine
=====================================================================
Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import cv2
import numpy as np
import pytest

# Ensure parent directory is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from groove_extractor import (
    isolate_muzzle_region,
    enhance_dark_grooves,
    binarize_grooves,
    skeletonize_grooves,
    create_groove_overlay,
    extract_muzzle_grooves_pipeline,
)


def _count_endpoints_and_junctions(skeleton: np.ndarray) -> Tuple[int, int]:
    """Helper to count endpoints and branch junctions in a 1-pixel-wide skeleton."""
    skel_binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) * skel_binary

    endpoints = int(np.sum((neighbor_count == 1) & (skel_binary == 1)))
    junctions = int(np.sum((neighbor_count >= 3) & (skel_binary == 1)))
    return endpoints, junctions


# -----------------------------------------------------------------------------
# Test 1: Synthetic dark line on light background produces detectable response
# -----------------------------------------------------------------------------
def test_synthetic_dark_line_detectable_response():
    """Verify that a dark line on a light background produces strong groove response."""
    h, w = 200, 200
    canvas = np.full((h, w, 3), 220, dtype=np.uint8)
    # Horizontal dark groove line in the center
    cv2.line(canvas, (40, 100), (160, 100), (30, 30, 30), thickness=3)

    mask = np.full((h, w), 255, dtype=np.uint8)

    # 1. Black-Hat enhancement
    bh_resp = enhance_dark_grooves(canvas, mask, method="blackhat", scales=[9, 15])
    assert bh_resp.dtype == np.uint8
    assert bh_resp.shape == (h, w)
    # The line region must have a strong peak response
    assert bh_resp[100, 100] > 150
    # Background region away from the line must have negligible response
    assert bh_resp[20, 20] < 20

    # 2. Frangi filter enhancement
    frangi_resp = enhance_dark_grooves(canvas, mask, method="frangi")
    assert frangi_resp.dtype == np.uint8
    assert frangi_resp[100, 100] > 150
    assert frangi_resp[20, 20] < 20

    # 3. Binarization & Skeletonization
    bin_mask = binarize_grooves(bh_resp, mask, threshold_method="otsu")
    assert bin_mask[100, 100] == 255
    assert bin_mask[20, 20] == 0

    skel = skeletonize_grooves(bin_mask)
    assert skel[100, 100] == 255
    # Total skeleton pixels should roughly match the line length (~120 pixels)
    skel_pixels = np.count_nonzero(skel)
    assert 100 <= skel_pixels <= 135


# -----------------------------------------------------------------------------
# Test 2: Synthetic branching line pattern (Y or T shape) retains branches
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("pattern_type", ["T", "Y"])
def test_synthetic_branching_line_retains_major_branches(pattern_type):
    """Verify skeletonization preserves multi-branch topology for Y and T shapes."""
    h, w = 240, 240
    canvas = np.full((h, w, 3), 220, dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)

    if pattern_type == "T":
        # T shape: stem + horizontal top bar
        cv2.line(canvas, (120, 70), (120, 170), (30, 30, 30), thickness=3)
        cv2.line(canvas, (60, 70), (180, 70), (30, 30, 30), thickness=3)
    else:  # Y shape
        # Y shape: vertical stem + two diagonal upper arms
        cv2.line(canvas, (120, 120), (120, 190), (30, 30, 30), thickness=3)
        cv2.line(canvas, (120, 120), (60, 50), (30, 30, 30), thickness=3)
        cv2.line(canvas, (120, 120), (180, 50), (30, 30, 30), thickness=3)

    result = extract_muzzle_grooves_pipeline(
        image_bgr=canvas,
        mask=mask,
        method="blackhat",
        scales=[9, 15],
        threshold_method="otsu",
        min_component_size=10,
    )

    skel = result["groove_skeleton"]
    assert np.count_nonzero(skel) > 80

    endpoints, junctions = _count_endpoints_and_junctions(skel)
    # A complete T or Y branch skeleton has at least 3 terminal endpoints and at least 1 branch node
    assert endpoints >= 3, f"Expected at least 3 endpoints for {pattern_type} pattern, got {endpoints}"
    assert junctions >= 1, f"Expected at least 1 junction for {pattern_type} pattern, got {junctions}"


# -----------------------------------------------------------------------------
# Test 3: Empty / zero mask produces empty groove mask and skeleton
# -----------------------------------------------------------------------------
def test_empty_zero_mask_produces_empty_groove_outputs():
    """Verify empty/zero mask gracefully outputs all-zero arrays without error."""
    h, w = 150, 150
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.line(canvas, (20, 75), (130, 75), (20, 20, 20), thickness=3)

    zero_mask = np.zeros((h, w), dtype=np.uint8)

    # Stage isolation
    clean_mask, masked_img = isolate_muzzle_region(canvas, mask=zero_mask)
    assert np.count_nonzero(clean_mask) == 0
    assert np.count_nonzero(masked_img) == 0

    # Enhancement
    bh_resp = enhance_dark_grooves(canvas, zero_mask, method="blackhat")
    assert np.count_nonzero(bh_resp) == 0

    frangi_resp = enhance_dark_grooves(canvas, zero_mask, method="frangi")
    assert np.count_nonzero(frangi_resp) == 0

    # Binarization
    groove_bin = binarize_grooves(bh_resp, zero_mask, threshold_method="otsu")
    assert np.count_nonzero(groove_bin) == 0

    # Skeletonization
    skel = skeletonize_grooves(groove_bin)
    assert np.count_nonzero(skel) == 0

    # Full pipeline with zero mask
    pipeline_res = extract_muzzle_grooves_pipeline(canvas, mask=zero_mask)
    assert pipeline_res["num_groove_pixels"] == 0
    assert pipeline_res["num_skeleton_pixels"] == 0
    assert np.count_nonzero(pipeline_res["groove_mask"]) == 0
    assert np.count_nonzero(pipeline_res["groove_skeleton"]) == 0


# -----------------------------------------------------------------------------
# Test 4: Mask boundary suppression: pixels outside mask are strictly 0
# -----------------------------------------------------------------------------
def test_mask_boundary_suppression():
    """Verify that any pixel outside the mask is strictly zeroed out."""
    h, w = 200, 200
    # Image with features everywhere across the whole canvas
    canvas = np.random.randint(50, 200, size=(h, w, 3), dtype=np.uint8)
    cv2.line(canvas, (10, 10), (190, 190), (0, 0, 0), thickness=3)

    # Restrict mask to a central circle only
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 40, 255, -1)

    outside = (mask == 0)

    # Enhancement
    enhanced = enhance_dark_grooves(canvas, mask, method="blackhat")
    assert np.all(enhanced[outside] == 0), "Non-zero enhanced pixels detected outside mask!"

    # Binarization
    binary = binarize_grooves(enhanced, mask, threshold_method="otsu")
    assert np.all(binary[outside] == 0), "Non-zero binarized pixels detected outside mask!"

    # Skeletonization
    skel = skeletonize_grooves(binary)
    assert np.all(skel[outside] == 0), "Non-zero skeleton pixels detected outside mask!"

    # Full pipeline
    pipeline_res = extract_muzzle_grooves_pipeline(canvas, mask=mask)
    assert np.all(pipeline_res["enhanced_response"][outside] == 0)
    assert np.all(pipeline_res["groove_mask"][outside] == 0)
    assert np.all(pipeline_res["groove_skeleton"][outside] == 0)


# -----------------------------------------------------------------------------
# Test 5: Uniform image does not crash and produces 0 grooves
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("val", [0, 128, 255])
@pytest.mark.parametrize("method", ["blackhat", "frangi"])
@pytest.mark.parametrize("threshold", ["otsu", "adaptive"])
def test_uniform_image_zero_grooves_and_no_crash(val, method, threshold):
    """Verify uniform images (solid black, solid gray, solid white) produce 0 grooves."""
    img = np.full((120, 120, 3), val, dtype=np.uint8)
    mask = np.full((120, 120), 255, dtype=np.uint8)

    res = extract_muzzle_grooves_pipeline(
        image_bgr=img,
        mask=mask,
        method=method,
        threshold_method=threshold,
    )

    assert res["num_groove_pixels"] == 0
    assert res["num_skeleton_pixels"] == 0
    assert np.count_nonzero(res["groove_mask"]) == 0
    assert np.count_nonzero(res["groove_skeleton"]) == 0


# -----------------------------------------------------------------------------
# Test 6: Different image resolutions work without crashing
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "h, w",
    [
        (112, 112),
        (224, 224),
        (448, 448),
        (600, 800),  # non-square
        (350, 500),  # arbitrary aspect ratio
    ],
)
def test_different_image_resolutions(h, w):
    """Verify pipeline runs smoothly across a spectrum of resolutions."""
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)
    # Draw a groove line scaled to resolution
    cv2.line(
        canvas,
        (w // 4, h // 2),
        (3 * w // 4, h // 2),
        (30, 30, 30),
        thickness=max(2, int(min(h, w) * 0.01)),
    )

    res = extract_muzzle_grooves_pipeline(
        image_bgr=canvas,
        method="blackhat",
        threshold_method="otsu",
    )

    assert res["resolution"] == (w, h)
    assert res["image_shape"] == (h, w, 3)
    assert res["clean_mask"].shape == (h, w)
    assert res["groove_skeleton"].shape == (h, w)
    assert res["num_skeleton_pixels"] > 0


# -----------------------------------------------------------------------------
# Test 7: Original input image is not modified in-place
# -----------------------------------------------------------------------------
def test_original_image_not_modified_in_place():
    """Verify input image is never mutated in any function or pipeline call."""
    img = np.full((150, 150, 3), 180, dtype=np.uint8)
    cv2.line(img, (30, 75), (120, 75), (40, 40, 40), 3)
    img_copy = img.copy()

    mask = np.full((150, 150), 255, dtype=np.uint8)
    mask_copy = mask.copy()

    # Call individual functions
    clean_mask, masked_img = isolate_muzzle_region(img, mask=mask)
    assert np.array_equal(img, img_copy), "isolate_muzzle_region mutated image_bgr!"
    assert np.array_equal(mask, mask_copy), "isolate_muzzle_region mutated mask!"

    enhanced = enhance_dark_grooves(img, mask)
    assert np.array_equal(img, img_copy), "enhance_dark_grooves mutated image_bgr!"

    bin_mask = binarize_grooves(enhanced, mask)
    skel = skeletonize_grooves(bin_mask)
    overlay = create_groove_overlay(img, skel)
    assert np.array_equal(img, img_copy), "create_groove_overlay mutated image_bgr!"

    # Call full pipeline
    pipeline_res = extract_muzzle_grooves_pipeline(img, mask=mask)
    assert np.array_equal(img, img_copy), "extract_muzzle_grooves_pipeline mutated image_bgr!"


# -----------------------------------------------------------------------------
# Test 8: Real muzzle crop test (using cow_muzzle_macro.jpg)
# -----------------------------------------------------------------------------
def test_real_muzzle_crop():
    """Verify groove extraction on a real cattle muzzle macro photograph."""
    sample_path = PROJECT_DIR / "test_samples" / "cow_muzzle_macro.jpg"
    assert sample_path.exists(), f"Sample image not found at {sample_path}"

    real_img = cv2.imread(str(sample_path))
    assert real_img is not None, "Failed to load cow_muzzle_macro.jpg"

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = extract_muzzle_grooves_pipeline(
            image_bgr=real_img,
            method="blackhat",
            scales=[9, 15, 21],
            threshold_method="otsu",
            output_dir=tmp_dir,
            prefix="test_cow",
        )

        assert res["timing_ms"] > 0.0
        assert res["num_groove_pixels"] > 1000
        assert res["num_skeleton_pixels"] > 500
        assert res["resolution"] == (real_img.shape[1], real_img.shape[0])

        # Verify saved artifacts
        saved = res["saved_paths"]
        assert Path(saved["groove_response"]).exists()
        assert Path(saved["groove_mask"]).exists()
        assert Path(saved["groove_skeleton"]).exists()
        assert Path(saved["groove_overlay"]).exists()

        # Check saved image dimensions
        saved_skel = cv2.imread(saved["groove_skeleton"], cv2.IMREAD_GRAYSCALE)
        assert saved_skel.shape == real_img.shape[:2]

        saved_over = cv2.imread(saved["groove_overlay"])
        assert saved_over.shape == real_img.shape


# -----------------------------------------------------------------------------
# Test 9: Nostril exclusion excludes dark upper cavity
# -----------------------------------------------------------------------------
def test_nostril_exclusion_behavior():
    """Verify exclude_nostrils removes dark cavity in upper region but keeps lower groove."""
    h, w = 300, 300
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)

    # Dark nostril cavity in upper muzzle region (y=80, x=150)
    cv2.ellipse(canvas, (150, 80), (35, 25), 0, 0, 360, (25, 25, 25), -1)
    # Dark groove line in lower region (y=200)
    cv2.line(canvas, (80, 200), (220, 200), (30, 30, 30), thickness=3)

    # With nostril exclusion
    clean_mask_ex, _ = isolate_muzzle_region(canvas, exclude_nostrils=True)
    # Nostril cavity center should be excluded
    assert clean_mask_ex[80, 150] == 0, "Nostril cavity was not excluded!"
    # Lower groove should be preserved
    assert clean_mask_ex[200, 150] == 255, "Lower groove was incorrectly excluded!"


# -----------------------------------------------------------------------------
# Test 10: Connected component noise pruning preserves elongated thin lines
# -----------------------------------------------------------------------------
def test_connected_component_noise_pruning():
    """Verify small noise blobs (< min_component_size) are pruned while short lines are retained."""
    h, w = 150, 150
    enhanced = np.zeros((h, w), dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)

    # Small compact noise blob: area ~ 9-12 pixels (< 15)
    cv2.circle(enhanced, (30, 30), 2, 250, -1)

    # Short thin groove line: length = 12 pixels, thickness = 1 (< 15 area)
    cv2.line(enhanced, (70, 70), (81, 70), 250, thickness=1)

    # Longer groove line: length = 30 pixels, thickness = 1 (> 15 area)
    cv2.line(enhanced, (100, 30), (100, 60), 250, thickness=1)

    binary = binarize_grooves(
        enhanced,
        mask,
        threshold_method="otsu",
        min_component_size=15,
    )

    # Noise blob should be removed
    assert binary[30, 30] == 0, "Noise blob was not pruned!"
    # Short line (elongated) should be retained despite area < 15
    assert binary[70, 75] == 255, "Short elongated line was pruned!"
    # Long line should be retained
    assert binary[45, 100] == 255, "Long line was pruned!"
