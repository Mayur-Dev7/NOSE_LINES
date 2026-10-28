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
import math
import cv2
import networkx as nx
import numpy as np
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


def prune_skeleton(
    groove_skeleton: np.ndarray,
    min_spur_length: int = 8,
    min_component_size: int = 30,
) -> np.ndarray:
    """
    Prunes spurs and small connected components from a 1-pixel binary skeleton.

    Parameters:
        groove_skeleton: 2D or 3D binary skeleton image (0 and 255 or 0 and 1).
        min_spur_length: Minimum pixel length for a spur branch connected to a junction.
                         Branches shorter than this connected to junctions are pruned.
        min_component_size: Minimum number of total 8-connected skeleton pixels.
                            Components smaller than this are erased.

    Returns:
        Pruned binary skeleton image as uint8 np.ndarray with values 0 and 255.
    """
    if groove_skeleton is None or groove_skeleton.size == 0:
        return np.zeros((0, 0), dtype=np.uint8) if groove_skeleton is None else np.zeros_like(groove_skeleton, dtype=np.uint8)

    if groove_skeleton.ndim == 3:
        groove_skeleton = cv2.cvtColor(groove_skeleton, cv2.COLOR_BGR2GRAY)

    skel = (groove_skeleton > 0).astype(np.uint8)
    if np.count_nonzero(skel) == 0:
        return (skel * 255).astype(np.uint8)

    # Initial pass: remove small disconnected components below min_component_size
    if min_component_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_component_size:
                skel[labels == lbl] = 0

    if np.count_nonzero(skel) == 0:
        return (skel * 255).astype(np.uint8)

    # Iterative spur pruning: remove degree-1 endpoints connected to nearest junction with length < min_spur_length
    H, W = skel.shape
    max_iter = 50

    for _ in range(max_iter):
        deg = compute_skeleton_neighbor_degrees(skel)
        endpoints = np.argwhere((deg == 1) & (skel == 1))
        junction_mask = (deg >= 3) & (skel == 1)

        if len(endpoints) == 0 or not np.any(junction_mask):
            break

        to_remove = set()
        for ey, ex in endpoints:
            path = [(int(ey), int(ex))]
            curr = (int(ey), int(ex))
            prev = None
            hit_junction = False

            while len(path) <= min_spur_length:
                nbrs = []
                for dy, dx in NEIGHBORS_8:
                    ny, nx = curr[0] + dy, curr[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] == 1 and (ny, nx) != prev:
                        nbrs.append((ny, nx))

                if not nbrs:
                    break

                junc_nbrs = [n for n in nbrs if junction_mask[n[0], n[1]]]
                if junc_nbrs:
                    hit_junction = True
                    break

                if len(nbrs) > 1:
                    break

                nxt = nbrs[0]
                path.append(nxt)
                prev = curr
                curr = nxt

            if hit_junction and len(path) < min_spur_length:
                for p in path:
                    to_remove.add(p)

        if not to_remove:
            break

        for py, px in to_remove:
            skel[py, px] = 0

        # Re-skeletonize to preserve 1-pixel thinness and clean up junction transitions
        skel = skeletonize(skel > 0).astype(np.uint8)

    # Final pass: remove any components that shrank below min_component_size
    if min_component_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_component_size:
                skel[labels == lbl] = 0

    return (skel * 255).astype(np.uint8)


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

    # Neighbor lookup helper
    def get_neighbors(py: int, px: int) -> List[Tuple[int, int]]:
        nbrs = []
        for dy, dx in NEIGHBORS_8:
            ny, nx = py + dy, px + dx
            if 0 <= ny < H and 0 <= nx < W and skel_binary[ny, nx] == 1:
                nbrs.append((ny, nx))
        return nbrs

    visited_path = np.zeros((H, W), dtype=bool)
    branches: List[Dict[str, Any]] = []

    def compute_and_add_branch(
        u: int,
        v: int,
        path_pixels: List[Tuple[int, int]]
    ) -> None:
        """Helper to compute branch metrics and record branch."""
        pos_u = nodes[u]["pos"]
        pos_v = nodes[v]["pos"]

        # Arc length: Euclidean distance along path including node centroids
        arc_len = 0.0
        p0 = path_pixels[0]
        arc_len += math.hypot(p0[1] - pos_u[0], p0[0] - pos_u[1])
        for i in range(len(path_pixels) - 1):
            arc_len += math.hypot(
                path_pixels[i + 1][1] - path_pixels[i][1],
                path_pixels[i + 1][0] - path_pixels[i][0]
            )
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
            orient = float(np.degrees(np.arctan2(dy, dx)))

        mid_idx = len(path_pixels) // 2
        mid_y, mid_x = path_pixels[mid_idx]

        branches.append({
            "start_node": u,
            "end_node": v,
            "pixel_coords": path_pixels,
            "arc_length": arc_len,
            "chord_length": chord_len,
            "curvature": curv,
            "orientation_deg": orient,
            "midpoint_norm": (float(mid_x / max(1, W)), float(mid_y / max(1, H))),
        })

    # 5. Branch Tracing: Step A (From Endpoints)
    for u in range(endpoint_count):
        ep_p = nodes[u]["pixels"][0]
        nbrs = get_neighbors(ep_p[0], ep_p[1])
        if not nbrs:
            continue
        q = nbrs[0]
        target_node = int(node_grid[q[0], q[1]])
        if target_node >= 0:
            # Direct neighbor between nodes (only add when u <= target_node to avoid duplication)
            if u <= target_node:
                compute_and_add_branch(u, target_node, [ep_p, q])
        else:
            if not visited_path[q[0], q[1]]:
                path = [ep_p, q]
                visited_path[q[0], q[1]] = True
                curr = q
                prev = ep_p
                v_node: Optional[int] = None
                while True:
                    cur_nbrs = get_neighbors(curr[0], curr[1])
                    cands = [n for n in cur_nbrs if n != prev]
                    hit_node = [n for n in cands if node_grid[n[0], n[1]] >= 0]
                    if hit_node:
                        end_p = hit_node[0]
                        v_node = int(node_grid[end_p[0], end_p[1]])
                        path.append(end_p)
                        break
                    unvis = [n for n in cands if not visited_path[n[0], n[1]]]
                    if not unvis:
                        break
                    nxt = unvis[0]
                    visited_path[nxt[0], nxt[1]] = True
                    path.append(nxt)
                    prev = curr
                    curr = nxt
                if v_node is not None:
                    compute_and_add_branch(u, v_node, path)

    # 5. Branch Tracing: Step B (From Junctions)
    for u in range(endpoint_count, endpoint_count + junction_count):
        u_pix_set = set(nodes[u]["pixels"])
        for p in nodes[u]["pixels"]:
            for q in get_neighbors(p[0], p[1]):
                if node_grid[q[0], q[1]] == -1 and not visited_path[q[0], q[1]]:
                    path = [p, q]
                    visited_path[q[0], q[1]] = True
                    curr = q
                    prev = p
                    v_node = None
                    while True:
                        cur_nbrs = get_neighbors(curr[0], curr[1])
                        cands = [n for n in cur_nbrs if n != prev]
                        hit_node = [
                            n for n in cands
                            if node_grid[n[0], n[1]] >= 0 and not (curr == q and n in u_pix_set)
                        ]
                        if hit_node:
                            end_p = hit_node[0]
                            v_node = int(node_grid[end_p[0], end_p[1]])
                            path.append(end_p)
                            break
                        unvis = [n for n in cands if not visited_path[n[0], n[1]] and n not in u_pix_set]
                        if not unvis:
                            # Check if looped back to u
                            u_hit = [n for n in cands if n in u_pix_set]
                            if u_hit:
                                end_p = u_hit[0]
                                v_node = u
                                path.append(end_p)
                            break
                        nxt = unvis[0]
                        visited_path[nxt[0], nxt[1]] = True
                        path.append(nxt)
                        prev = curr
                        curr = nxt
                    if v_node is not None:
                        compute_and_add_branch(u, v_node, path)

    # 5. Branch Tracing: Step C (Pure closed loops with no endpoints and no junctions)
    skel_pts = np.argwhere(skel_binary == 1)
    for y, x in skel_pts:
        if node_grid[y, x] == -1 and not visited_path[y, x]:
            loop_p = [(int(y), int(x))]
            visited_path[y, x] = True
            curr = (int(y), int(x))
            prev = None
            while True:
                nbrs = [n for n in get_neighbors(curr[0], curr[1]) if n != prev]
                unvis = [n for n in nbrs if not visited_path[n[0], n[1]]]
                if unvis:
                    nxt = unvis[0]
                    visited_path[nxt[0], nxt[1]] = True
                    loop_p.append(nxt)
                    prev = curr
                    curr = nxt
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
            arc_len = 0.0
            for i in range(len(loop_cycle) - 1):
                arc_len += math.hypot(
                    loop_cycle[i + 1][1] - loop_cycle[i][1],
                    loop_cycle[i + 1][0] - loop_cycle[i][0]
                )
            mid_idx = len(loop_p) // 2
            mid_y, mid_x = loop_p[mid_idx]

            branches.append({
                "start_node": loop_id,
                "end_node": loop_id,
                "pixel_coords": loop_cycle,
                "arc_length": arc_len,
                "chord_length": 0.0,
                "curvature": arc_len,
                "orientation_deg": 0.0,
                "midpoint_norm": (float(mid_x / max(1, W)), float(mid_y / max(1, H))),
            })

    # Minutiae orientation for nodes
    for nid, ndata in nodes.items():
        ntype = ndata["type"]
        pos = ndata["pos"]
        node_branches = [b for b in branches if b["start_node"] == nid or b["end_node"] == nid]

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
                orient = float(np.degrees(np.arctan2(dy, dx))) if (abs(dx) > 1e-4 or abs(dy) > 1e-4) else 0.0
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
                        ang = float(np.degrees(np.arctan2(dy, dx)))
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
