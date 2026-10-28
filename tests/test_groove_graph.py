"""
Unit and Integration Tests for Cattle Muzzle Groove Graph Extraction
====================================================================
Tests topological graph extraction, node identification, junction clustering,
branch tracing, curvature analysis, cycle counting, NetworkX graph integrity,
and visualization.

Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

import os
import sys
import math
from pathlib import Path

# Ensure project root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import networkx as nx
import numpy as np
import pytest
from skimage.morphology import skeletonize

from groove_graph import (
    extract_skeleton_graph,
    visualize_groove_graph,
    compute_skeleton_neighbor_degrees,
    cluster_junction_pixels,
    prune_skeleton,
)
from groove_extractor import extract_muzzle_grooves_pipeline


# -----------------------------------------------------------------------------
# Test 1: Single straight line skeleton produces 2 endpoints, 1 branch, 0 junctions, curvature ~1.0
# -----------------------------------------------------------------------------
def test_single_straight_line_topology():
    """Verify single straight line produces 2 endpoints, 1 branch, 0 junctions, curvature ~1.0."""
    h, w = 60, 60
    skel = np.zeros((h, w), dtype=np.uint8)
    # Horizontal line from (25, 10) to (25, 50) -> length = 40 pixels
    skel[25, 10:51] = 255

    res = extract_skeleton_graph(skel)

    assert res["endpoint_count"] == 2
    assert res["junction_count"] == 0
    assert len(res["branches"]) == 1
    assert len(res["nodes"]) == 2

    branch = res["branches"][0]
    assert branch["chord_length"] == pytest.approx(40.0, rel=1e-3)
    assert branch["arc_length"] == pytest.approx(40.0, rel=1e-3)
    assert branch["curvature"] == pytest.approx(1.0, rel=1e-3)
    assert branch["orientation_deg"] == pytest.approx(0.0, abs=1e-1)
    assert branch["midpoint_norm"][1] == pytest.approx(25.0 / h, abs=0.05)
    assert branch["midpoint_norm"][0] == pytest.approx(30.0 / w, abs=0.05)

    # Topological consistency
    assert res["connected_component_count"] == 1
    assert res["cycle_count"] == 0
    assert res["euler_characteristic"] == 1  # V - E = 2 - 1 = 1


# -----------------------------------------------------------------------------
# Test 2: Y-shaped skeleton produces 3 endpoints, 1 junction, 3 branches
# -----------------------------------------------------------------------------
def test_y_shaped_skeleton_topology():
    """Verify Y-shaped skeleton produces 3 endpoints, 1 junction, 3 branches."""
    h, w = 80, 80
    canvas = np.zeros((h, w), dtype=np.uint8)
    # 3 arms radiating from (40, 40)
    cv2.line(canvas, (40, 40), (15, 15), 255, 1)
    cv2.line(canvas, (40, 40), (65, 15), 255, 1)
    cv2.line(canvas, (40, 40), (40, 70), 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    res = extract_skeleton_graph(skel)

    assert res["endpoint_count"] == 3
    assert res["junction_count"] == 1
    assert len(res["branches"]) == 3
    assert len(res["nodes"]) == 4

    # All 3 branches should connect an endpoint to the single central junction
    junc_id = [n["id"] for n in res["node_list"] if n["type"] == "junction"][0]
    ep_ids = set(n["id"] for n in res["node_list"] if n["type"] == "endpoint")

    for branch in res["branches"]:
        u, v = branch["start_node"], branch["end_node"]
        assert (u == junc_id and v in ep_ids) or (v == junc_id and u in ep_ids)
        # Each arm is straight, curvature should be ~1.0
        assert branch["curvature"] == pytest.approx(1.0, rel=0.1)

    assert res["cycle_count"] == 0
    assert res["connected_component_count"] == 1
    assert res["euler_characteristic"] == 1  # V - E = 4 - 3 = 1


# -----------------------------------------------------------------------------
# Test 3: T-shaped skeleton produces 3 endpoints, 1 junction, 3 branches
# -----------------------------------------------------------------------------
def test_t_shaped_skeleton_topology():
    """Verify T-shaped skeleton produces 3 endpoints, 1 junction, 3 branches."""
    h, w = 80, 80
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Horizontal top bar
    cv2.line(canvas, (15, 25), (65, 25), 255, 1)
    # Vertical stem
    cv2.line(canvas, (40, 25), (40, 70), 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    res = extract_skeleton_graph(skel)

    assert res["endpoint_count"] == 3
    assert res["junction_count"] == 1
    assert len(res["branches"]) == 3
    assert len(res["nodes"]) == 4

    # Curvature of straight arms should be ~1.0
    for branch in res["branches"]:
        assert branch["curvature"] == pytest.approx(1.0, rel=0.1)

    assert res["cycle_count"] == 0
    assert res["connected_component_count"] == 1
    assert res["euler_characteristic"] == 1  # V - E = 4 - 3 = 1


# -----------------------------------------------------------------------------
# Test 4: Closed loop (circle / square) produces 0 endpoints, cycle_count = 1
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["circle", "square"])
def test_closed_loop_topology(shape):
    """Verify closed loop produces 0 endpoints, cycle_count = 1, euler_char = 0."""
    h, w = 100, 100
    canvas = np.zeros((h, w), dtype=np.uint8)

    if shape == "circle":
        cv2.circle(canvas, (50, 50), 30, 255, 1)
    else:
        cv2.rectangle(canvas, (30, 30), (70, 70), 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    res = extract_skeleton_graph(skel)

    assert res["endpoint_count"] == 0
    assert res["cycle_count"] == 1
    assert res["connected_component_count"] == 1
    assert res["euler_characteristic"] == 0  # V - E = 1 - 1 = 0
    assert len(res["branches"]) == 1

    loop_branch = res["branches"][0]
    assert loop_branch["start_node"] == loop_branch["end_node"]
    assert loop_branch["chord_length"] == 0.0
    assert loop_branch["arc_length"] > 100.0


# -----------------------------------------------------------------------------
# Test 5: Empty skeleton (all zeros) returns 0 nodes, 0 branches without crash
# -----------------------------------------------------------------------------
def test_empty_skeleton_zero_outputs_no_crash():
    """Verify all-zero skeleton returns 0 nodes, 0 branches, and 0 descriptors without crashing."""
    empty_skel = np.zeros((64, 64), dtype=np.uint8)
    res = extract_skeleton_graph(empty_skel)

    assert res["skeleton_pixel_count"] == 0
    assert res["skeleton_density"] == 0.0
    assert res["endpoint_count"] == 0
    assert res["junction_count"] == 0
    assert res["connected_component_count"] == 0
    assert res["cycle_count"] == 0
    assert res["euler_characteristic"] == 0
    assert len(res["nodes"]) == 0
    assert len(res["branches"]) == 0
    assert res["mean_branch_length"] == 0.0
    assert res["spatial_centroid"] == (0.0, 0.0)
    assert res["spatial_dispersion"] == 0.0
    assert isinstance(res["graph"], nx.Graph)
    assert res["graph"].number_of_nodes() == 0
    assert res["graph"].number_of_edges() == 0


# -----------------------------------------------------------------------------
# Test 6: NetworkX graph export matches topological properties
# -----------------------------------------------------------------------------
def test_networkx_graph_export_topology():
    """Verify NetworkX graph object attributes and algebraic topology descriptors."""
    h, w = 120, 120
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Figure: two T junctions connected by a horizontal line, plus a circle
    cv2.line(canvas, (20, 40), (100, 40), 255, 1)
    cv2.line(canvas, (40, 40), (40, 80), 255, 1)
    cv2.line(canvas, (80, 40), (80, 80), 255, 1)
    # Disconnected circle
    cv2.circle(canvas, (60, 100), 12, 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    res = extract_skeleton_graph(skel)
    G = res["graph"]

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(res["nodes"])

    # Node attributes check
    for nid, data in G.nodes(data=True):
        assert "pos" in data
        assert isinstance(data["pos"], tuple)
        assert len(data["pos"]) == 2
        assert "type" in data
        assert data["type"] in ("endpoint", "junction", "loop")

    # Edge attributes check
    for u, v, data in G.edges(data=True):
        assert "length" in data
        assert data["length"] > 0.0
        assert "curvature" in data
        assert data["curvature"] >= 0.8
        assert "orientation" in data
        assert -180.0 <= data["orientation"] <= 180.0

    # Topological invariant checks
    assert nx.number_connected_components(G) == res["connected_component_count"]
    assert len(nx.cycle_basis(G)) == res["cycle_count"]
    assert (G.number_of_nodes() - G.number_of_edges()) == res["euler_characteristic"]


# -----------------------------------------------------------------------------
# Test 7: Real sample muzzle skeleton extracts rich graph structure (> 20 nodes, > 10 branches)
# -----------------------------------------------------------------------------
def test_real_muzzle_sample_rich_graph():
    """Verify real cattle muzzle image produces rich graph topology (> 20 nodes, > 10 branches)."""
    samples_dir = Path(__file__).resolve().parent.parent / "test_samples"
    sample_file = samples_dir / "cow_video_frame_1.jpg"
    if not sample_file.exists():
        sample_file = samples_dir / "cow_muzzle_macro.jpg"

    assert sample_file.exists(), f"Sample image {sample_file} not found!"

    img = cv2.imread(str(sample_file))
    assert img is not None, "Failed to load sample muzzle image"

    pipeline_res = extract_muzzle_grooves_pipeline(img, method="blackhat")
    skel = pipeline_res["groove_skeleton"]
    mask = pipeline_res["groove_mask"]

    res = extract_skeleton_graph(skel, mask=mask)

    assert res["skeleton_pixel_count"] > 500
    assert res["skeleton_density"] > 0.0
    assert len(res["nodes"]) > 20, f"Expected >20 nodes, got {len(res['nodes'])}"
    assert len(res["branches"]) > 10, f"Expected >10 branches, got {len(res['branches'])}"
    assert res["endpoint_count"] > 10
    assert res["junction_count"] > 5
    assert res["connected_component_count"] >= 1
    assert res["mean_branch_length"] > 0.0
    assert res["spatial_dispersion"] > 0.0

    # Verify NetworkX graph consistency
    G = res["graph"]
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(res["nodes"])


# -----------------------------------------------------------------------------
# Test 8: Visualization overlay generation
# -----------------------------------------------------------------------------
def test_visualize_groove_graph_overlay():
    """Verify visualize_groove_graph produces valid non-destructive BGR overlay."""
    h, w = 100, 100
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.line(canvas, (20, 50), (80, 50), 255, 1)
    cv2.line(canvas, (50, 20), (50, 80), 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    graph_data = extract_skeleton_graph(skel)

    bgr = np.full((h, w, 3), 40, dtype=np.uint8)
    bgr_copy = bgr.copy()

    overlay = visualize_groove_graph(bgr, graph_data)

    # Must return new array without modifying input
    assert overlay is not bgr
    np.testing.assert_array_equal(bgr, bgr_copy)

    assert overlay.shape == (h, w, 3)
    assert overlay.dtype == np.uint8
    # Must have colored pixels drawn
    assert not np.array_equal(overlay, bgr)


# -----------------------------------------------------------------------------
# Test 9: Curved line curvature is strictly greater than 1.0
# -----------------------------------------------------------------------------
def test_curved_arc_curvature():
    """Verify curved semicircle branch produces curvature significantly > 1.0."""
    h, w = 120, 120
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Semicircle arc: center (60, 60), radius 30, from 0 to 180 degrees
    cv2.ellipse(canvas, (60, 60), (30, 30), 0, 0, 180, 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    res = extract_skeleton_graph(skel)

    assert res["endpoint_count"] == 2
    assert res["junction_count"] == 0
    assert len(res["branches"]) == 1

    b = res["branches"][0]
    # Semicircle arc length = pi * R ~ 94.2
    # Chord length = 2 * R = 60.0
    # Curvature = arc / chord ~ pi / 2 ~ 1.57 > 1.0
    assert b["chord_length"] == pytest.approx(60.0, abs=4.0)
    assert b["curvature"] > 1.4
    assert b["curvature"] < 1.8


# -----------------------------------------------------------------------------
# Test 10: Skeleton Pruning removes short spurs < min_spur_length and small components
# -----------------------------------------------------------------------------
def test_prune_skeleton_removes_short_spurs_and_small_components():
    """Verify prune_skeleton eliminates spurs shorter than min_spur_length and small components."""
    h, w = 80, 80
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Long backbone: length ~51 px
    cv2.line(canvas, (15, 40), (65, 40), 255, 1)
    # Short spur: length ~5 px (attached at (40, 40), goes to (40, 35))
    cv2.line(canvas, (40, 40), (40, 35), 255, 1)
    # Disconnected small component: length ~12 px
    cv2.line(canvas, (10, 10), (22, 10), 255, 1)

    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)

    # Before pruning, graph should have 4 endpoints (2 from backbone, 1 from spur, 2 from small comp)
    res_raw = extract_skeleton_graph(skel, prune=False)
    assert res_raw["endpoint_count"] >= 4
    assert res_raw["junction_count"] >= 1

    # Prune with min_spur_length=8 and min_component_size=30
    pruned = prune_skeleton(skel, min_spur_length=8, min_component_size=30)
    res_pruned = extract_skeleton_graph(pruned, prune=False)

    # The spur is gone, the small component is gone -> exactly 2 endpoints, 0 junctions, 1 branch!
    assert res_pruned["endpoint_count"] == 2
    assert res_pruned["junction_count"] == 0
    assert len(res_pruned["branches"]) == 1
    assert res_pruned["connected_component_count"] == 1


# -----------------------------------------------------------------------------
# Test 11: Minutiae orientation on endpoints and junctions
# -----------------------------------------------------------------------------
def test_minutiae_orientation_endpoints_and_junctions():
    """Verify minutiae nodes have mathematically consistent orientation attributes."""
    h, w = 80, 80
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Horizontal line from (20, 40) to (60, 40)
    cv2.line(canvas, (20, 40), (60, 40), 255, 1)
    skel = (skeletonize(canvas > 0) * 255).astype(np.uint8)

    res = extract_skeleton_graph(skel)
    for n in res["node_list"]:
        assert "orientation" in n
        assert isinstance(n["orientation"], float)
        assert -180.0 <= n["orientation"] <= 180.0

    # Left endpoint should point right (dx > 0, angle ~0 deg)
    left_node = [n for n in res["node_list"] if n["pos"][0] < 30][0]
    right_node = [n for n in res["node_list"] if n["pos"][0] > 50][0]
    assert left_node["orientation"] == pytest.approx(0.0, abs=5.0)
    # Right endpoint should point left (dx < 0, angle ~180 or -180 deg)
    assert abs(right_node["orientation"]) == pytest.approx(180.0, abs=5.0)

    # Check NetworkX G nodes also contain orientation
    G = res["graph"]
    for nid, d in G.nodes(data=True):
        assert "orientation" in d
        assert -180.0 <= d["orientation"] <= 180.0

