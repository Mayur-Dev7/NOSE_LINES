"""
Biometric Groove Graph and Topological Feature Extraction Module
===============================================================
Extracts mathematically rigorous graph representations (nodes, junctions,
endpoints, branches, curvature, cycles, Euler characteristic) from 1-pixel
binary skeletons of cattle and buffalo muzzle groove patterns.

Compatible with OpenCV, scikit-image, NetworkX, and NumPy.
Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import math
import cv2
import networkx as nx
import numpy as np
import numba
from skimage.morphology import skeletonize



# 8-connectivity neighbor offsets: (dy, dx)
NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# 3x3 convolution kernel to calculate 8-neighbor degrees
DEGREE_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)


def compute_skeleton_neighbor_degrees(skel_binary: np.ndarray) -> np.ndarray:
    """
    Computes 8-neighbor connectivity degree for each skeleton pixel.

    Degree interpretation on 1-pixel skeletons:
      - 0: Isolated skeleton pixel
      - 1: Endpoint
      - 2: Regular path pixel
      - >= 3: Junction / bifurcation candidate
    """
    return cv2.filter2D(skel_binary, cv2.CV_16S, DEGREE_KERNEL) * skel_binary


def cluster_junction_pixels(
    junc_mask: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clusters adjacent / contiguous junction pixels using 8-connected components.

    Returns:
        num_labels: Total connected component labels (including 0 background)
        labels: 2D label map (1 to num_labels - 1)
        stats: Component bounding boxes and areas
        centroids: Floating-point centroid coordinates (cx, cy)
    """
    return cv2.connectedComponentsWithStats(junc_mask, connectivity=8)


@numba.njit(cache=True)
def _crossing_number(skel: np.ndarray, py: int, px: int) -> int:
    """Computes the Rutovitz crossing number around pixel (py, px) in a binary skeleton."""
    H, W = skel.shape
    # offsets in clockwise order: (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
    v0 = 1 if (py > 0 and skel[py - 1, px] > 0) else 0
    v1 = 1 if (py > 0 and px + 1 < W and skel[py - 1, px + 1] > 0) else 0
    v2 = 1 if (px + 1 < W and skel[py, px + 1] > 0) else 0
    v3 = 1 if (py + 1 < H and px + 1 < W and skel[py + 1, px + 1] > 0) else 0
    v4 = 1 if (py + 1 < H and skel[py + 1, px] > 0) else 0
    v5 = 1 if (py + 1 < H and px > 0 and skel[py + 1, px - 1] > 0) else 0
    v6 = 1 if (px > 0 and skel[py, px - 1] > 0) else 0
    v7 = 1 if (py > 0 and px > 0 and skel[py - 1, px - 1] > 0) else 0

    cn = 0
    if v0 == 0 and v1 == 1: cn += 1
    if v1 == 0 and v2 == 1: cn += 1
    if v2 == 0 and v3 == 1: cn += 1
    if v3 == 0 and v4 == 1: cn += 1
    if v4 == 0 and v5 == 1: cn += 1
    if v5 == 0 and v6 == 1: cn += 1
    if v6 == 0 and v7 == 1: cn += 1
    if v7 == 0 and v0 == 1: cn += 1
    return cn


@numba.njit(cache=True)
def _prune_spurs_pass_numba(skel: np.ndarray, min_spur_length: int) -> bool:
    """
    Numba-compiled single pass of endpoint spur pruning on a binary skeleton.
    Scans in exact row-major order, walks branches towards junctions, and clears
    spurs shorter than or equal to min_spur_length.
    """
    H, W = skel.shape
    # NEIGHBORS_8 order: (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    dy_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dx_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

    # 1. Collect all endpoints (skel > 0 and 8-neighbor sum == 1) in row-major order
    ep_count = 0
    for y in range(H):
        for x in range(W):
            if skel[y, x] > 0:
                deg = 0
                for d in range(8):
                    ny = y + dy_arr[d]
                    nx = x + dx_arr[d]
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0:
                        deg += 1
                if deg == 1:
                    ep_count += 1

    if ep_count == 0:
        return False

    ep_y = np.empty(ep_count, dtype=np.int32)
    ep_x = np.empty(ep_count, dtype=np.int32)
    idx = 0
    for y in range(H):
        for x in range(W):
            if skel[y, x] > 0:
                deg = 0
                for d in range(8):
                    ny = y + dy_arr[d]
                    nx = x + dx_arr[d]
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0:
                        deg += 1
                if deg == 1:
                    ep_y[idx] = y
                    ep_x[idx] = x
                    idx += 1

    removed_any = False
    max_path_buf = min_spur_length + 2
    path_y = np.empty(max_path_buf, dtype=np.int32)
    path_x = np.empty(max_path_buf, dtype=np.int32)

    for i in range(ep_count):
        ey = ep_y[i]
        ex = ep_x[i]
        # In case this endpoint was already cleared by a previously pruned spur in this pass
        if skel[ey, ex] == 0:
            continue

        path_y[0] = ey
        path_x[0] = ex
        path_len = 1

        prev_y = -1
        prev_x = -1
        curr_y = ey
        curr_x = ex
        reached_junction = False

        while True:
            n_count = 0
            nxt_y = -1
            nxt_x = -1
            for d in range(8):
                ny = curr_y + dy_arr[d]
                nx = curr_x + dx_arr[d]
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0 and (ny != prev_y or nx != prev_x):
                    n_count += 1
                    nxt_y = ny
                    nxt_x = nx

            if n_count != 1:
                reached_junction = (n_count >= 2) or (_crossing_number(skel, curr_y, curr_x) >= 3)
                break

            if _crossing_number(skel, nxt_y, nxt_x) >= 3:
                reached_junction = True
                break

            if path_len < max_path_buf:
                path_y[path_len] = nxt_y
                path_x[path_len] = nxt_x
                path_len += 1
            else:
                path_len += 1

            prev_y = curr_y
            prev_x = curr_x
            curr_y = nxt_y
            curr_x = nxt_x

            if path_len > min_spur_length:
                break

        if reached_junction and path_len <= min_spur_length:
            for p in range(path_len):
                skel[path_y[p], path_x[p]] = 0
            removed_any = True

    return removed_any


def prune_skeleton(
    groove_skeleton: np.ndarray,
    min_spur_length: int = 8,
    min_component_size: int = 30,
) -> np.ndarray:
    """Remove small components and short endpoint branches; return uint8 values {0,255}."""
    if groove_skeleton is None or groove_skeleton.size == 0:
        return np.zeros((0, 0), dtype=np.uint8) if groove_skeleton is None else np.zeros_like(groove_skeleton, dtype=np.uint8)
    if groove_skeleton.ndim == 3:
        groove_skeleton = cv2.cvtColor(groove_skeleton, cv2.COLOR_BGR2GRAY)
    skel = (groove_skeleton > 0).astype(np.uint8)
    if not np.any(skel):
        return (skel * 255).astype(np.uint8)

    def remove_small_components(binary):
        if min_component_size <= 0:
            return binary
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < min_component_size:
                binary[labels == label] = 0
        return binary

    skel = remove_small_components(skel)
    H, W = skel.shape

    # Prune one endpoint branch at a time using Numba-compiled walking loop.
    for _ in range(max(H, W)):
        removed_any = _prune_spurs_pass_numba(skel, int(min_spur_length))
        if not removed_any:
            break
        skel = remove_small_components(skel)
    return (skel * 255).astype(np.uint8)


def extract_minutiae_points(skel, max_points=250, max_search_radius=80):
    """Return endpoints/junctions ranked by longest incident skeleton branch with bounded fast tracing."""
    if skel is None or skel.size == 0:
        return []
    binary = (np.asarray(skel) > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        binary = (binary > 0).astype(np.uint8)
    degrees = compute_skeleton_neighbor_degrees(binary)
    node_mask = (binary > 0) & ((degrees == 1) | (degrees >= 3))
    nodes_yx = np.argwhere(node_mask)
    if len(nodes_yx) == 0:
        return []
    H, W = binary.shape

    # Zero-pad binary and node_mask by 1px for fast C-like neighbor lookups without bounds checks
    pad_bin = np.pad(binary, 1, mode="constant", constant_values=0)
    pad_node = np.pad(node_mask, 1, mode="constant", constant_values=False)
    deltas = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    records = []
    for yv, xv in nodes_yx:
        y, x = int(yv), int(xv)
        deg = degrees[y, x]
        kind = "endpoint" if deg == 1 else "junction"
        py, px = y + 1, x + 1

        longest = 0.0
        best_angle = 0.0

        for dy, dx in deltas:
            if pad_bin[py + dy, px + dx]:
                prev_y, prev_x = py, px
                curr_y, curr_x = py + dy, px + dx
                branch_len = 1
                branch_angle = math.degrees(math.atan2(float(dy), float(dx))) % 360.0

                while branch_len < max_search_radius and not pad_node[curr_y, curr_x]:
                    next_y, next_x = -1, -1
                    for ddy, ddx in deltas:
                        ny, nx = curr_y + ddy, curr_x + ddx
                        if (ny != prev_y or nx != prev_x) and pad_bin[ny, nx]:
                            if next_y != -1:
                                # Multiple branches/junction encountered
                                next_y = -1
                                break
                            next_y, next_x = ny, nx
                    if next_y == -1:
                        break
                    prev_y, prev_x = curr_y, curr_x
                    curr_y, curr_x = next_y, next_x
                    branch_len += 1

                if branch_len > longest:
                    longest = float(branch_len)
                    best_angle = branch_angle

        records.append({
            "x": float(x),
            "y": float(y),
            "type": kind,
            "angle": best_angle,
            "weight": longest,
        })
    records.sort(key=lambda item: (-item["weight"], item["y"], item["x"]))
    return records[:max(0, int(max_points))]



def extract_skeleton_graph(
    groove_skeleton: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_spur_length: int = 8,
    min_component_size: int = 30,
    prune: bool = True,
) -> Dict[str, Any]:
    """
    Extracts topological graph representation and global descriptors from a 1-pixel binary skeleton.

    Parameters:
        groove_skeleton: 1-pixel binary skeleton image (uint8 0 or 255).
        mask: Optional binary ROI mask (uint8 0 or 255) for accurate density calculation.
        min_spur_length: Minimum spur length for pruning (default 8).
        min_component_size: Minimum connected component size for pruning (default 30).
        prune: Whether to prune spurs and small components before graph extraction (default True).

    Returns:
        Dict containing:
          - 'graph': NetworkX simple Graph (G = nx.Graph())
          - 'nodes': Dict[int, Dict[str, Any]] mapping node_id to node attributes (including 'orientation')
          - 'node_list': List[Dict[str, Any]]
          - 'branches': List[Dict[str, Any]] extracted branch metadata
          - 'skeleton_pixel_count': Total skeleton pixels
          - 'skeleton_density': skeleton_pixels / total_mask_pixels
          - 'endpoint_count': Count of degree-1 nodes
          - 'junction_count': Count of clustered junction nodes
          - 'connected_component_count': Number of connected graph components
          - 'cycle_count': Number of independent cycles / closed loops
          - 'euler_characteristic': V - E
          - 'mean_branch_length': Mean arc length across branches
          - 'median_branch_length': Median arc length across branches
          - 'std_branch_length': Standard deviation of branch lengths
          - 'curvature_mean': Mean curvature across branches
          - 'curvature_std': Standard deviation of curvature across branches
          - 'spatial_centroid': Normalized coordinate (mean_x / W, mean_y / H)
          - 'spatial_dispersion': Radius of gyration of skeleton pixels
          - 'descriptors': Dictionary containing all global descriptors
    """
    if prune and groove_skeleton is not None and groove_skeleton.size > 0:
        groove_skeleton = prune_skeleton(
            groove_skeleton,
            min_spur_length=min_spur_length,
            min_component_size=min_component_size,
        )

    if groove_skeleton is None or groove_skeleton.size == 0:
        empty_g = nx.Graph()
        empty_desc = {
            "skeleton_pixel_count": 0,
            "skeleton_density": 0.0,
            "endpoint_count": 0,
            "junction_count": 0,
            "branch_count": 0,
            "connected_component_count": 0,
            "cycle_count": 0,
            "euler_characteristic": 0,
            "mean_branch_length": 0.0,
            "median_branch_length": 0.0,
            "std_branch_length": 0.0,
            "curvature_mean": 0.0,
            "curvature_std": 0.0,
            "spatial_centroid": (0.0, 0.0),
            "spatial_dispersion": 0.0,
        }
        res = {
            "graph": empty_g,
            "nodes": {},
            "node_list": [],
            "branches": [],
            "descriptors": empty_desc,
        }
        res.update(empty_desc)
        return res

    # Ensure 2D uint8 binary image (0 or 1)
    if groove_skeleton.ndim == 3:
        groove_skeleton = cv2.cvtColor(groove_skeleton, cv2.COLOR_BGR2GRAY)

    skel_binary = (groove_skeleton > 0).astype(np.uint8)
    H, W = skel_binary.shape
    skel_pixel_count = int(np.count_nonzero(skel_binary))

    # Calculate total mask pixels for density normalization
    if mask is not None and mask.size > 0:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        total_mask_pixels = int(np.count_nonzero(mask > 0))
    else:
        total_mask_pixels = H * W

    if total_mask_pixels <= 0:
        total_mask_pixels = max(1, H * W)

    skel_density = float(skel_pixel_count / max(1, total_mask_pixels))

    # Fast return for empty skeleton
    if skel_pixel_count == 0:
        empty_g = nx.Graph()
        empty_desc = {
            "skeleton_pixel_count": 0,
            "skeleton_density": 0.0,
            "endpoint_count": 0,
            "junction_count": 0,
            "branch_count": 0,
            "connected_component_count": 0,
            "cycle_count": 0,
            "euler_characteristic": 0,
            "mean_branch_length": 0.0,
            "median_branch_length": 0.0,
            "std_branch_length": 0.0,
            "curvature_mean": 0.0,
            "curvature_std": 0.0,
            "spatial_centroid": (0.0, 0.0),
            "spatial_dispersion": 0.0,
        }
        res = {
            "graph": empty_g,
            "nodes": {},
            "node_list": [],
            "branches": [],
            "descriptors": empty_desc,
        }
        res.update(empty_desc)
        return res

    # 1. Neighbor degree computation
    deg = compute_skeleton_neighbor_degrees(skel_binary)

    endpoint_mask = ((deg == 1) & (skel_binary == 1)).astype(np.uint8)
    junc_mask = ((deg >= 3) & (skel_binary == 1)).astype(np.uint8)

    # 2. Junction clustering via 8-connected components
    num_junc, junc_labels, stats, centroids = cluster_junction_pixels(junc_mask)
    junction_count = max(0, num_junc - 1)

    node_grid = np.full((H, W), -1, dtype=np.int32)
    nodes: Dict[int, Dict[str, Any]] = {}
    next_node_id = 0

    # 3. Node identification: Endpoints
    ep_coords = np.argwhere(endpoint_mask == 1)
    endpoint_count = len(ep_coords)
    for y, x in ep_coords:
        p = (int(y), int(x))
        node_grid[p[0], p[1]] = next_node_id
        nodes[next_node_id] = {
            "id": next_node_id,
            "pos": (float(x), float(y)),
            "type": "endpoint",
            "pixels": [p],
            "degree": 0,
        }
        next_node_id += 1

    # 4. Node identification: Clustered Junctions
    for k in range(1, num_junc):
        cx, cy = float(centroids[k][0]), float(centroids[k][1])
        nodes[next_node_id] = {
            "id": next_node_id,
            "pos": (cx, cy),
            "type": "junction",
            "pixels": [],
            "degree": 0,
        }
        next_node_id += 1

    junc_pixels = np.argwhere(junc_labels > 0)
    for y, x in junc_pixels:
        lbl = int(junc_labels[y, x])
        nid = endpoint_count + (lbl - 1)
        node_grid[y, x] = nid
        nodes[nid]["pixels"].append((int(y), int(x)))

    # Zero-pad arrays by 1px for fast neighbor lookups without bounds tests
    pad_skel = np.pad(skel_binary, 1, mode="constant", constant_values=0)
    pad_node_grid = np.pad(node_grid, 1, mode="constant", constant_values=-1)
    visited_path = np.zeros((H + 2, W + 2), dtype=bool)

    branches: List[Dict[str, Any]] = []
    node_to_branches: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    deltas = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def compute_and_add_branch(
        u: int,
        v: int,
        path_pixels: List[Tuple[int, int]]
    ) -> None:
        """Helper to compute branch metrics and record branch."""
        pos_u = nodes[u]["pos"]
        pos_v = nodes[v]["pos"]

        # Arc length: Euclidean distance along path including node centroids
        p0 = path_pixels[0]
        arc_len = math.hypot(p0[1] - pos_u[0], p0[0] - pos_u[1])
        if len(path_pixels) > 1:
            diffs = np.diff(np.array(path_pixels, dtype=np.float32), axis=0)
            arc_len += float(np.sum(np.hypot(diffs[:, 1], diffs[:, 0])))
        p_last = path_pixels[-1]
        arc_len += math.hypot(pos_v[0] - p_last[1], pos_v[1] - p_last[0])

        chord_len = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
        if u == v:
            chord_len = 0.0
            curv = arc_len
            orient = 0.0
        else:
            curv = arc_len / max(1.0, chord_len)
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            orient = float(math.degrees(math.atan2(dy, dx)))

        mid_idx = len(path_pixels) // 2
        mid_y, mid_x = path_pixels[mid_idx]

        b_record = {
            "start_node": u,
            "end_node": v,
            "pixel_coords": path_pixels,
            "arc_length": arc_len,
            "chord_length": chord_len,
            "curvature": curv,
            "orientation_deg": orient,
            "midpoint_norm": (float(mid_x / max(1, W)), float(mid_y / max(1, H))),
        }
        branches.append(b_record)
        node_to_branches[u].append(b_record)
        if u != v:
            node_to_branches[v].append(b_record)

    # 5. Branch Tracing: Step A (From Endpoints)
    for u in range(endpoint_count):
        ey, ex = nodes[u]["pixels"][0]
        py, px = ey + 1, ex + 1
        q_pos = None
        for dy, dx in deltas:
            if pad_skel[py + dy, px + dx]:
                q_pos = (py + dy, px + dx)
                break
        if q_pos is None:
            continue

        tgt = pad_node_grid[q_pos[0], q_pos[1]]
        q_orig = (q_pos[0] - 1, q_pos[1] - 1)
        if tgt >= 0:
            if u <= tgt:
                compute_and_add_branch(u, int(tgt), [(ey, ex), q_orig])
        else:
            if not visited_path[q_pos[0], q_pos[1]]:
                visited_path[q_pos[0], q_pos[1]] = True
                path = [(ey, ex), q_orig]
                curr_y, curr_x = q_pos
                prev_y, prev_x = py, px
                v_node: Optional[int] = None
                while True:
                    next_y, next_x = -1, -1
                    hit_tgt = -1
                    for dy, dx in deltas:
                        cny, cnx = curr_y + dy, curr_x + dx
                        if (cny != prev_y or cnx != prev_x) and pad_skel[cny, cnx]:
                            ng = pad_node_grid[cny, cnx]
                            if ng >= 0:
                                hit_tgt = ng
                                next_y, next_x = cny, cnx
                                break
                            elif not visited_path[cny, cnx]:
                                next_y, next_x = cny, cnx
                    if hit_tgt >= 0:
                        path.append((next_y - 1, next_x - 1))
                        v_node = int(hit_tgt)
                        break
                    if next_y == -1:
                        break
                    visited_path[next_y, next_x] = True
                    path.append((next_y - 1, next_x - 1))
                    prev_y, prev_x = curr_y, curr_x
                    curr_y, curr_x = next_y, next_x
                if v_node is not None:
                    compute_and_add_branch(u, v_node, path)

    # 5. Branch Tracing: Step B (From Junctions)
    for u in range(endpoint_count, endpoint_count + junction_count):
        u_pix_set = {(p[0] + 1, p[1] + 1) for p in nodes[u]["pixels"]}
        for ey, ex in nodes[u]["pixels"]:
            py, px = ey + 1, ex + 1
            for dy, dx in deltas:
                qy, qx = py + dy, px + dx
                if pad_skel[qy, qx] and pad_node_grid[qy, qx] == -1 and not visited_path[qy, qx]:
                    visited_path[qy, qx] = True
                    path = [(ey, ex), (qy - 1, qx - 1)]
                    curr_y, curr_x = qy, qx
                    prev_y, prev_x = py, px
                    v_node = None
                    while True:
                        next_y, next_x = -1, -1
                        hit_tgt = -1
                        hit_u = False
                        for ddy, ddx in deltas:
                            cny, cnx = curr_y + ddy, curr_x + ddx
                            if (cny != prev_y or cnx != prev_x) and pad_skel[cny, cnx]:
                                ng = pad_node_grid[cny, cnx]
                                if ng >= 0 and not (curr_y == qy and curr_x == qx and (cny, cnx) in u_pix_set):
                                    hit_tgt = ng
                                    next_y, next_x = cny, cnx
                                    break
                                elif not visited_path[cny, cnx] and (cny, cnx) not in u_pix_set:
                                    next_y, next_x = cny, cnx
                                elif (cny, cnx) in u_pix_set:
                                    hit_u = True
                                    next_y, next_x = cny, cnx
                        if hit_tgt >= 0:
                            path.append((next_y - 1, next_x - 1))
                            v_node = int(hit_tgt)
                            break
                        if next_y == -1:
                            if hit_u:
                                path.append((next_y - 1, next_x - 1))
                                v_node = u
                            break
                        visited_path[next_y, next_x] = True
                        path.append((next_y - 1, next_x - 1))
                        prev_y, prev_x = curr_y, curr_x
                        curr_y, curr_x = next_y, next_x
                    if v_node is not None:
                        compute_and_add_branch(u, v_node, path)

    # 5. Branch Tracing: Step C (Pure closed loops with no endpoints and no junctions)
    skel_pts = np.argwhere(skel_binary == 1)
    for y, x in skel_pts:
        py, px = y + 1, x + 1
        if pad_node_grid[py, px] == -1 and not visited_path[py, px]:
            loop_p = [(int(y), int(x))]
            visited_path[py, px] = True
            curr_y, curr_x = py, px
            prev_y, prev_x = -1, -1
            while True:
                next_y, next_x = -1, -1
                for dy, dx in deltas:
                    cny, cnx = curr_y + dy, curr_x + dx
                    if (cny != prev_y or cnx != prev_x) and pad_skel[cny, cnx] and not visited_path[cny, cnx]:
                        next_y, next_x = cny, cnx
                        break
                if next_y != -1:
                    visited_path[next_y, next_x] = True
                    loop_p.append((next_y - 1, next_x - 1))
                    prev_y, prev_x = curr_y, curr_x
                    curr_y, curr_x = next_y, next_x
                else:
                    break

            cx = float(np.mean([p[1] for p in loop_p]))
            cy = float(np.mean([p[0] for p in loop_p]))
            loop_id = next_node_id
            next_node_id += 1
            nodes[loop_id] = {
                "id": loop_id,
                "pos": (cx, cy),
                "type": "loop",
                "pixels": loop_p,
                "degree": 0,
            }

            loop_cycle = loop_p + [loop_p[0]]
            diffs = np.diff(np.array(loop_cycle, dtype=np.float32), axis=0)
            arc_len = float(np.sum(np.hypot(diffs[:, 1], diffs[:, 0])))
            mid_idx = len(loop_p) // 2
            mid_y, mid_x = loop_p[mid_idx]

            b_record = {
                "start_node": loop_id,
                "end_node": loop_id,
                "pixel_coords": loop_cycle,
                "arc_length": arc_len,
                "chord_length": 0.0,
                "curvature": arc_len,
                "orientation_deg": 0.0,
                "midpoint_norm": (float(mid_x / max(1, W)), float(mid_y / max(1, H))),
            }
            branches.append(b_record)
            node_to_branches[loop_id].append(b_record)

    # Minutiae orientation for nodes using fast node_to_branches lookup (O(1))
    for nid, ndata in nodes.items():
        ntype = ndata["type"]
        pos = ndata["pos"]
        node_branches = node_to_branches[nid]

        if ntype == "endpoint":
            if node_branches:
                b = node_branches[0]
                coords = b["pixel_coords"]
                if b["start_node"] == nid:
                    k = min(4, len(coords) - 1)
                    target_p = coords[k]
                else:
                    k = max(0, len(coords) - 1 - min(4, len(coords) - 1))
                    target_p = coords[k]
                dx = target_p[1] - pos[0]
                dy = target_p[0] - pos[1]
                orient = float(math.degrees(math.atan2(dy, dx))) if (abs(dx) > 1e-4 or abs(dy) > 1e-4) else 0.0
            else:
                orient = 0.0
            ndata["orientation"] = round(orient, 2)

        elif ntype == "junction":
            if node_branches:
                out_angles = []
                out_lens = []
                for b in node_branches:
                    coords = b["pixel_coords"]
                    if b["start_node"] == nid:
                        k = min(4, len(coords) - 1)
                        target_p = coords[k]
                    else:
                        k = max(0, len(coords) - 1 - min(4, len(coords) - 1))
                        target_p = coords[k]
                    dx = target_p[1] - pos[0]
                    dy = target_p[0] - pos[1]
                    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                        ang = float(math.degrees(math.atan2(dy, dx)))
                        out_angles.append(ang)
                        out_lens.append(b["arc_length"])

                if len(out_angles) == 3:
                    min_diff = 999.0
                    fork_pair = (0, 1)
                    for i in range(3):
                        for j in range(i + 1, 3):
                            diff = abs((out_angles[i] - out_angles[j] + 180.0) % 360.0 - 180.0)
                            if diff < min_diff:
                                min_diff = diff
                                fork_pair = (i, j)
                    trunk_idx = [idx for idx in range(3) if idx not in fork_pair][0]
                    orient = out_angles[trunk_idx]
                elif out_angles:
                    best_idx = int(np.argmax(out_lens))
                    orient = out_angles[best_idx]
                else:
                    orient = 0.0
            else:
                orient = 0.0
            ndata["orientation"] = round(orient, 2)

        else:
            ndata["orientation"] = 0.0

    # 6. Build NetworkX Graph G
    G = nx.Graph()
    for nid, ndata in nodes.items():
        G.add_node(nid, pos=ndata["pos"], type=ndata["type"], orientation=ndata["orientation"])

    for b in branches:
        u, v = b["start_node"], b["end_node"]
        G.add_edge(
            u, v,
            length=b["arc_length"],
            arc_length=b["arc_length"],
            chord_length=b["chord_length"],
            curvature=b["curvature"],
            orientation=b["orientation_deg"],
            orientation_deg=b["orientation_deg"],
            pixel_coords=b["pixel_coords"],
            midpoint_norm=b["midpoint_norm"],
        )

    # Update degrees on node dicts
    for nid in nodes:
        if nid in G:
            nodes[nid]["degree"] = int(G.degree[nid])

    # 7. Compute Global Numerical & Topological Descriptors
    branch_lens = [b["arc_length"] for b in branches]
    curvs = [b["curvature"] for b in branches]

    mean_branch_len = float(np.mean(branch_lens)) if branch_lens else 0.0
    median_branch_len = float(np.median(branch_lens)) if branch_lens else 0.0
    std_branch_len = float(np.std(branch_lens)) if branch_lens else 0.0

    curvature_mean = float(np.mean(curvs)) if curvs else 0.0
    curvature_std = float(np.std(curvs)) if curvs else 0.0

    all_y = skel_pts[:, 0]
    all_x = skel_pts[:, 1]
    mean_y = float(np.mean(all_y))
    mean_x = float(np.mean(all_x))
    spatial_centroid = (float(mean_x / max(1, W)), float(mean_y / max(1, H)))
    spatial_dispersion = float(np.sqrt(np.mean((all_x - mean_x) ** 2 + (all_y - mean_y) ** 2)))

    conn_comp_count = int(nx.number_connected_components(G)) if G.number_of_nodes() > 0 else 0
    cycle_count = int(len(nx.cycle_basis(G))) if G.number_of_nodes() > 0 else 0
    euler_characteristic = int(G.number_of_nodes() - G.number_of_edges())

    descriptors = {
        "skeleton_pixel_count": skel_pixel_count,
        "skeleton_density": skel_density,
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
        "branch_count": len(branches),
        "connected_component_count": conn_comp_count,
        "cycle_count": cycle_count,
        "euler_characteristic": euler_characteristic,
        "mean_branch_length": mean_branch_len,
        "median_branch_length": median_branch_len,
        "std_branch_length": std_branch_len,
        "curvature_mean": curvature_mean,
        "curvature_std": curvature_std,
        "spatial_centroid": spatial_centroid,
        "spatial_dispersion": spatial_dispersion,
    }

    result: Dict[str, Any] = {
        "graph": G,
        "nodes": nodes,
        "node_list": list(nodes.values()),
        "branches": branches,
        "descriptors": descriptors,
    }
    result.update(descriptors)
    return result


def visualize_groove_graph(
    image_bgr: np.ndarray,
    graph_data: Dict[str, Any],
    branch_color: Tuple[int, int, int] = (255, 255, 0),     # Cyan in BGR
    endpoint_color: Tuple[int, int, int] = (0, 255, 0),     # Green in BGR
    junction_color: Tuple[int, int, int] = (0, 0, 255),     # Red in BGR
    branch_thickness: int = 1,
    node_radius: int = 3,
) -> np.ndarray:
    """
    Renders colored overlay of the extracted graph on top of the image.

    Colors:
      - Branches: Cyan / Blue lines
      - Endpoints: Green circles
      - Junctions: Red / Magenta circles

    Returns:
        vis: 3-channel BGR visualization image (np.ndarray uint8).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    if image_bgr.ndim == 2:
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    else:
        vis = image_bgr.copy()

    # 1. Draw branches
    branches = graph_data.get("branches", [])
    for branch in branches:
        coords = branch.get("pixel_coords", [])
        if len(coords) >= 2:
            pts = np.array([(int(x), int(y)) for (y, x) in coords], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                vis,
                [pts],
                isClosed=False,
                color=branch_color,
                thickness=branch_thickness,
                lineType=cv2.LINE_AA,
            )

    # 2. Draw nodes
    nodes = graph_data.get("nodes", {})
    node_items = nodes.values() if isinstance(nodes, dict) else nodes
    for nd in node_items:
        ntype = nd.get("type", "")
        pos = nd.get("pos", (0.0, 0.0))
        pt = (int(round(pos[0])), int(round(pos[1])))

        if ntype == "endpoint":
            cv2.circle(
                vis,
                pt,
                radius=node_radius,
                color=endpoint_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                vis,
                pt,
                radius=node_radius + 1,
                color=(0, 160, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif ntype == "junction":
            cv2.circle(
                vis,
                pt,
                radius=node_radius + 1,
                color=junction_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                vis,
                pt,
                radius=node_radius + 2,
                color=(0, 0, 160),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return vis


def _warmup_numba():
    """Warms up numba njit functions at module import with dummy arrays to eliminate initial JIT latency."""
    try:
        dummy = np.zeros((32, 32), dtype=np.uint8)
        dummy[16, 8:24] = 1
        dummy[8:24, 16] = 1
        _crossing_number(dummy, 16, 16)
        _prune_spurs_pass_numba(dummy, 4)
    except Exception:
        pass


_warmup_numba()
