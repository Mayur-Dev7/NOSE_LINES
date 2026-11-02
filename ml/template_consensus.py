"""
Consensus Analysis, Spatial Alignment, and Feature Correspondence Module
========================================================================
Builds consolidated multi-image reference templates for cattle muzzle biometrics
via spatial alignment assessment, branch feature correspondence, robust median/MAD
statistics, and overall topological stability scoring.
Strict scope: /home/adis_mayur/F_CODES/nose_lines
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid
import numpy as np

from template_schema import (
    SCHEMA_VERSION,
    robust_median_and_mad,
    calculate_stability,
    make_empty_template,
)

# Standard scalar descriptors tracked for multi-image biometric consensus
STANDARD_DESCRIPTORS = [
    "skeleton_pixel_count",
    "skeleton_density",
    "endpoint_count",
    "junction_count",
    "connected_component_count",
    "cycle_count",
    "mean_branch_length",
    "euler_characteristic",
    "curvature_mean",
    "quality_score",
]

TOPOLOGICAL_DESCRIPTOR_KEYS = [
    "skeleton_pixel_count",
    "skeleton_density",
    "endpoint_count",
    "junction_count",
    "connected_component_count",
    "cycle_count",
    "mean_branch_length",
    "euler_characteristic",
    "curvature_mean",
]


def _compute_axial_median(angles_deg: List[float]) -> float:
    """
    Computes directional median/mean for axial line data (angles in [0, 180) deg).
    Because lines have 180-degree symmetry (undirected lines), angles are doubled to 2*theta,
    averaged via vector addition (sum cos, sum sin), and halved back to [0, 180) deg.
    """
    if not angles_deg:
        return 0.0
    if len(angles_deg) == 1:
        return float(angles_deg[0] % 180.0)

    # If angle spread is compact without wrapping around 0/180 boundary
    angles_arr = np.array(angles_deg, dtype=np.float64) % 180.0
    if np.max(angles_arr) - np.min(angles_arr) < 90.0:
        return float(np.median(angles_arr))

    rad = np.radians(angles_arr * 2.0)
    c = float(np.sum(np.cos(rad)))
    s = float(np.sum(np.sin(rad)))
    mean_2rad = np.arctan2(s, c)
    mean_deg = (np.degrees(mean_2rad) / 2.0) % 180.0
    return float(mean_deg)


def _normalize_branch(b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts normalized midpoint (x, y) in [0, 1], orientation_deg in [0, 180),
    and length from heterogeneous branch dictionary formats.
    """
    # 1. Midpoint in [0, 1] normalized coordinates
    if "midpoint" in b:
        mp = b["midpoint"]
    elif "midpoint_norm" in b:
        mp = b["midpoint_norm"]
    elif "centroid_norm" in b:
        mp = b["centroid_norm"]
    elif "centroid" in b:
        mp = b["centroid"]
    elif "start" in b and "end" in b:
        s, e = b["start"], b["end"]
        mp = [(s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0]
    elif "p1" in b and "p2" in b:
        s, e = b["p1"], b["p2"]
        mp = [(s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0]
    else:
        mp = [0.5, 0.5]

    # 2. Length
    if "length" in b:
        length = float(b["length"])
    elif "branch_length" in b:
        length = float(b["branch_length"])
    elif ("start" in b and "end" in b) or ("p1" in b and "p2" in b):
        s = b.get("start", b.get("p1"))
        e = b.get("end", b.get("p2"))
        length = float(np.hypot(e[0] - s[0], e[1] - s[1]))
    else:
        length = 0.1

    # 3. Orientation in [0, 180) deg
    if "orientation_deg" in b:
        ori = float(b["orientation_deg"]) % 180.0
    elif "orientation" in b:
        ori = float(b["orientation"]) % 180.0
    elif "angle_deg" in b:
        ori = float(b["angle_deg"]) % 180.0
    elif "angle" in b:
        ori = float(b["angle"]) % 180.0
    elif ("start" in b and "end" in b) or ("p1" in b and "p2" in b):
        s = b.get("start", b.get("p1"))
        e = b.get("end", b.get("p2"))
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        ori = float(np.degrees(np.arctan2(dy, dx))) % 180.0
    else:
        ori = 0.0

    return {
        "midpoint": [float(mp[0]), float(mp[1])],
        "orientation_deg": float(ori),
        "length": float(length),
        "raw": b,
    }


def _branches_match(
    b1: Dict[str, Any],
    b2: Dict[str, Any],
    reliable_alignment: bool = True,
) -> bool:
    """
    Determines if two branches match according to the feature correspondence contract:
    - Midpoint distance <= 0.12 in normalized coords [0, 1] x [0, 1] (when alignment reliable)
    - Orientation difference <= 30 deg
    - Length ratio between 0.6 and 1.6
    """
    # 1. Orientation difference <= 30 deg
    diff_angle = abs(b1["orientation_deg"] - b2["orientation_deg"])
    diff_angle = min(diff_angle, 180.0 - diff_angle)
    if diff_angle > 30.0:
        return False

    # 2. Length ratio between 0.6 and 1.6
    l1 = max(float(b1["length"]), 1e-6)
    l2 = max(float(b2["length"]), 1e-6)
    ratio = min(l1, l2) / max(l1, l2)
    if ratio < 0.6:
        return False

    # 3. Midpoint distance <= 0.12 (when alignment reliable)
    if reliable_alignment:
        d_mid = float(
            np.hypot(
                b1["midpoint"][0] - b2["midpoint"][0],
                b1["midpoint"][1] - b2["midpoint"][1],
            )
        )
        if d_mid > 0.12:
            return False
    else:
        # Fallback spatial constraint when alignment is uncertain
        d_mid = float(
            np.hypot(
                b1["midpoint"][0] - b2["midpoint"][0],
                b1["midpoint"][1] - b2["midpoint"][1],
            )
        )
        if d_mid > 0.25:
            return False

    return True


class _RecurringStructureCluster:
    """Maintains a cluster of matched branches representing a single recurring ridge/groove."""

    def __init__(self, branch: Dict[str, Any], image_index: int):
        self.branches: List[Dict[str, Any]] = [branch]
        self.image_indices: set = {image_index}
        self.med_x: float = branch["midpoint"][0]
        self.med_y: float = branch["midpoint"][1]
        self.med_len: float = branch["length"]
        self.med_ori: float = branch["orientation_deg"]

    def matches(self, branch: Dict[str, Any], reliable_alignment: bool) -> bool:
        cluster_rep = {
            "midpoint": [self.med_x, self.med_y],
            "orientation_deg": self.med_ori,
            "length": self.med_len,
        }
        return _branches_match(cluster_rep, branch, reliable_alignment=reliable_alignment)

    def distance_to(self, branch: Dict[str, Any]) -> float:
        return float(
            np.hypot(
                branch["midpoint"][0] - self.med_x,
                branch["midpoint"][1] - self.med_y,
            )
        )

    def add_branch(self, branch: Dict[str, Any], image_index: int):
        self.branches.append(branch)
        self.image_indices.add(image_index)
        xs = [b["midpoint"][0] for b in self.branches]
        ys = [b["midpoint"][1] for b in self.branches]
        lens = [b["length"] for b in self.branches]
        self.med_x = float(np.median(xs))
        self.med_y = float(np.median(ys))
        self.med_len = float(np.median(lens))
        self.med_ori = _compute_axial_median([b["orientation_deg"] for b in self.branches])


def _extract_recurring_structures(
    independent_records: List[Dict[str, Any]],
    usable_count: int,
    reliable_alignment: bool,
    min_recurrence_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Groups matched branches across independent usable images into recurring ridge/groove structures.
    Filters by minimum recurrence ratio (k / N_usable >= min_recurrence_ratio).
    """
    if usable_count == 0 or not independent_records:
        return []

    clusters: List[_RecurringStructureCluster] = []

    for img_idx, rec in enumerate(independent_records):
        branches_raw = rec.get("branches", [])
        if not branches_raw:
            continue
        parsed_branches = [_normalize_branch(b) for b in branches_raw]

        for b in parsed_branches:
            # Find eligible clusters that do not yet have a branch from this image
            best_cluster = None
            best_dist = float("inf")

            for c in clusters:
                if img_idx in c.image_indices:
                    continue
                if c.matches(b, reliable_alignment=reliable_alignment):
                    d = c.distance_to(b)
                    if d < best_dist:
                        best_dist = d
                        best_cluster = c

            if best_cluster is not None:
                best_cluster.add_branch(b, img_idx)
            else:
                clusters.append(_RecurringStructureCluster(b, img_idx))

    # Evaluate recurrence ratio and filter
    recurring_structures = []
    for c in clusters:
        k = len(c.image_indices)
        rec_ratio = k / usable_count
        # When multi-image batch, require at least 2 observations
        if usable_count > 1 and k < 2:
            continue
        if rec_ratio >= min_recurrence_ratio:
            recurring_structures.append({
                "cluster": c,
                "observed_count": k,
                "recurrence_ratio": round(float(rec_ratio), 3),
            })

    # Sort descending by observed count, then recurrence ratio
    recurring_structures.sort(
        key=lambda item: (-item["observed_count"], -item["recurrence_ratio"])
    )

    results = []
    for idx, item in enumerate(recurring_structures):
        c = item["cluster"]
        results.append({
            "structure_id": f"recurring_ridge_{idx + 1:02d}",
            "recurrence_ratio": item["recurrence_ratio"],
            "observed_count": item["observed_count"],
            "centroid_norm": [round(c.med_x, 3), round(c.med_y, 3)],
            "median_length": round(c.med_len, 2),
            "median_orientation_deg": round(c.med_ori, 2),
        })

    return results


def _assess_spatial_registration(
    usable_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates spatial registration feasibility across usable muzzle images
    using normalized centroid offsets and principal axis orientation differences.
    """
    n_usable = len(usable_records)
    if n_usable < 2:
        return {
            "registration_attempted": False,
            "reliable_alignment": False,
            "method": "none",
            "quality_metric": 0.0,
            "reason": "At least 2 usable views required for registration assessment.",
        }

    # Check for explicit precomputed alignment metrics if provided
    explicit_metrics = [
        float(r["alignment_quality"])
        for r in usable_records
        if "alignment_quality" in r and r["alignment_quality"] is not None
    ]
    if len(explicit_metrics) == n_usable:
        quality_metric = round(float(np.mean(explicit_metrics)), 3)
        reliable = bool(quality_metric >= 0.70)
        return {
            "registration_attempted": True,
            "reliable_alignment": reliable,
            "method": "precomputed",
            "quality_metric": quality_metric,
            "reason": f"Explicit registration metric average: {quality_metric:.3f}",
        }

    centroids = []
    orientations = []

    for r in usable_records:
        # Centroid extraction
        c = None
        if "centroid_norm" in r and r["centroid_norm"] is not None:
            c = r["centroid_norm"]
        elif "centroid" in r and r["centroid"] is not None:
            c = r["centroid"]
        elif "mask" in r and isinstance(r["mask"], np.ndarray) and r["mask"].size > 0:
            m = r["mask"]
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                c = [float(np.mean(xs)) / m.shape[1], float(np.mean(ys)) / m.shape[0]]
        elif "branches" in r and len(r.get("branches", [])) > 0:
            parsed = [_normalize_branch(b) for b in r["branches"]]
            c = [
                float(np.mean([b["midpoint"][0] for b in parsed])),
                float(np.mean([b["midpoint"][1] for b in parsed])),
            ]
        if c is None:
            c = [0.5, 0.5]
        centroids.append([float(c[0]), float(c[1])])

        # Principal axis orientation extraction
        ori = None
        if "principal_axis_deg" in r and r["principal_axis_deg"] is not None:
            ori = float(r["principal_axis_deg"]) % 180.0
        elif "orientation_deg" in r and r["orientation_deg"] is not None:
            ori = float(r["orientation_deg"]) % 180.0
        elif "orientation" in r and r["orientation"] is not None:
            ori = float(r["orientation"]) % 180.0
        elif "mask" in r and isinstance(r["mask"], np.ndarray) and r["mask"].size > 0:
            m = r["mask"]
            ys, xs = np.where(m > 0)
            if len(xs) > 4:
                mx, my = np.mean(xs), np.mean(ys)
                mu20 = np.sum((xs - mx) ** 2)
                mu02 = np.sum((ys - my) ** 2)
                mu11 = np.sum((xs - mx) * (ys - my))
                ori = (0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))) % 180.0
        elif "branches" in r and len(r.get("branches", [])) > 0:
            parsed = [_normalize_branch(b) for b in r["branches"]]
            ori = _compute_axial_median([b["orientation_deg"] for b in parsed])
        if ori is None:
            ori = 0.0
        orientations.append(float(ori))

    # Compute pairwise deviations
    c_dists = []
    ori_dists = []
    for i in range(n_usable):
        for j in range(i + 1, n_usable):
            dc = float(
                np.hypot(
                    centroids[i][0] - centroids[j][0],
                    centroids[i][1] - centroids[j][1],
                )
            )
            c_dists.append(dc)
            d_ori = abs(orientations[i] - orientations[j])
            d_ori = min(d_ori, 180.0 - d_ori)
            ori_dists.append(d_ori)

    mean_dc = float(np.mean(c_dists)) if c_dists else 0.0
    mean_d_ori = float(np.mean(ori_dists)) if ori_dists else 0.0

    # Alignment quality score calculation
    score_c = max(0.0, 1.0 - mean_dc / 0.20)
    score_ori = max(0.0, 1.0 - mean_d_ori / 45.0)

    quality_metric = round(float(0.6 * score_c + 0.4 * score_ori), 3)
    reliable = bool(quality_metric >= 0.70)

    if reliable:
        reason = (
            f"Consistent spatial registration (centroid offset={mean_dc:.3f}, "
            f"principal axis delta={mean_d_ori:.1f} deg)"
        )
    else:
        reason = (
            f"Registration uncertain (score {quality_metric:.2f} < 0.70): "
            f"centroid offset={mean_dc:.3f}, principal axis delta={mean_d_ori:.1f} deg"
        )

    return {
        "registration_attempted": True,
        "reliable_alignment": reliable,
        "method": "centroid_principal_axis",
        "quality_metric": quality_metric,
        "reason": reason,
    }


def _extract_scalar(rec: Dict[str, Any], key: str) -> Optional[float]:
    """Helper to extract a scalar numeric descriptor across flat or nested record structures."""
    # Check top-level
    if key in rec and isinstance(rec[key], (int, float)) and not isinstance(rec[key], bool):
        return float(rec[key])

    # Check graph_descriptors
    g = rec.get("graph_descriptors", {})
    if isinstance(g, dict) and key in g and isinstance(g[key], (int, float)) and not isinstance(g[key], bool):
        return float(g[key])

    # Check quality
    q = rec.get("quality", {})
    if isinstance(q, dict) and key in q and isinstance(q[key], (int, float)) and not isinstance(q[key], bool):
        return float(q[key])

    # Check descriptors
    d = rec.get("descriptors", {})
    if isinstance(d, dict) and key in d and isinstance(d[key], (int, float)) and not isinstance(d[key], bool):
        return float(d[key])

    return None


def _mark_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies duplicate or burst frames in the batch and sets is_duplicate flag.
    Examines explicit duplicate flags, duplicate source_ids, and duplicate_of references.
    """
    processed = []
    seen_ids = set()

    for idx, r in enumerate(records):
        rec_copy = dict(r)
        is_dup = bool(rec_copy.get("is_duplicate", False))

        src_id = rec_copy.get("source_id")
        if src_id is not None:
            if src_id in seen_ids:
                is_dup = True
            else:
                seen_ids.add(src_id)

        if rec_copy.get("duplicate_of") is not None:
            is_dup = True

        rec_copy["is_duplicate"] = is_dup
        processed.append(rec_copy)

    return processed


def build_consensus_template(
    batch_records: List[Dict[str, Any]],
    template_id: Optional[str] = None,
    min_recurrence_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Builds a canonical multi-image reference consensus template from per-image batch records.

    Parameters
    ----------
    batch_records : List[Dict[str, Any]]
        Per-image processing records (quality metrics, usable status, duplicate flag,
        graph descriptors, branches).
    template_id : Optional[str]
        Unique reference template identifier. If omitted, a timestamped ID is generated.
    min_recurrence_ratio : float
        Minimum recurrence fraction (k / N_usable) required for a branch to be classified
        as a recurring reference structure.

    Returns
    -------
    Dict[str, Any]
        Complete consolidated reference template adhering to template_schema.py.
    """
    if template_id is None:
        template_id = f"tmpl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    submitted_count = len(batch_records)
    template = make_empty_template(template_id=template_id, submitted_count=submitted_count)
    template["processing_config"] = {
        "min_recurrence_ratio": min_recurrence_ratio,
        "schema_version": SCHEMA_VERSION,
    }

    # Step 1: Filter Usable & Independent Images
    marked_records = _mark_duplicates(batch_records)

    usable_records = [r for r in marked_records if r.get("is_usable", True)]
    usable_count = len(usable_records)

    independent_records = [r for r in usable_records if not r.get("is_duplicate", False)]
    independent_view_count = len(independent_records)

    rejected_count = submitted_count - usable_count
    duplicate_count = sum(1 for r in marked_records if r.get("is_duplicate", False))
    usable_ratio = round(usable_count / submitted_count, 3) if submitted_count > 0 else 0.0

    # Populate source images and per-image features
    source_images = []
    per_image_features = []

    for idx, rec in enumerate(marked_records):
        img_idx = rec.get("image_index", idx)
        q = rec.get("quality", {})
        if not isinstance(q, dict):
            q = {}

        src_entry = {
            "image_index": img_idx,
            "source_id": rec.get("source_id", f"img_{img_idx}"),
            "is_usable": bool(rec.get("is_usable", True)),
            "is_duplicate": bool(rec.get("is_duplicate", False)),
            "quality": {
                "quality_score": float(q.get("quality_score", rec.get("quality_score", 0.0))),
                "sharpness": float(q.get("sharpness", rec.get("sharpness", 0.0))),
                "contrast": float(q.get("contrast", rec.get("contrast", 0.0))),
                "valid_area_pct": float(q.get("valid_area_pct", rec.get("valid_area_pct", 100.0))),
            },
            "rejection_reasons": list(rec.get("rejection_reasons", [])),
        }
        source_images.append(src_entry)

        # Graph descriptors
        g_desc = {}
        if "graph_descriptors" in rec and isinstance(rec["graph_descriptors"], dict):
            g_desc.update(rec["graph_descriptors"])
        for k in STANDARD_DESCRIPTORS:
            if k not in g_desc:
                val = _extract_scalar(rec, k)
                if val is not None:
                    g_desc[k] = val

        feat_entry = {
            "image_index": img_idx,
            "graph_descriptors": g_desc,
            "branches": rec.get("branches", []),
        }
        per_image_features.append(feat_entry)

    template["source_images"] = source_images
    template["per_image_features"] = per_image_features

    # Quality Summary stats
    quality_scores = [s["quality"]["quality_score"] for s in source_images if s["is_usable"]]
    sharpness_scores = [s["quality"]["sharpness"] for s in source_images if s["is_usable"]]

    mean_quality = round(float(np.mean(quality_scores)), 3) if quality_scores else 0.0
    median_sharpness = round(float(np.median(sharpness_scores)), 3) if sharpness_scores else 0.0

    template["batch_summary"] = {
        "submitted_count": submitted_count,
        "processed_count": submitted_count,
        "usable_count": usable_count,
        "independent_view_count": independent_view_count,
        "rejected_count": rejected_count,
    }

    template["quality_summary"] = {
        "mean_quality_score": mean_quality,
        "median_sharpness": median_sharpness,
        "usable_ratio": usable_ratio,
        "duplicate_count": duplicate_count,
    }

    # Handle Usable Count = 0 edge case
    if usable_count == 0:
        template["confidence_level"] = "insufficient_data"
        template["feature_stability"] = {
            "overall_stability_score": 0.0,
            "confidence_level": "insufficient_data",
        }
        template["warnings"].append(
            "No usable images in batch (all images rejected). Consensus template cannot be generated."
        )
        template["usable_count"] = 0
        template["submitted_count"] = submitted_count
        template["independent_view_count"] = 0
        template["confidence_level"] = "insufficient_data"
        template["overall_stability_score"] = 0.0
        return template

    # Add single view warning if applicable
    if independent_view_count == 1:
        template["warnings"].append(
            "Template has limited multi-image evidence (only 1 independent usable view)."
        )

    # Step 2: Alignment Assessment
    alignment_summary = _assess_spatial_registration(usable_records)
    template["alignment_summary"] = alignment_summary
    reliable_alignment = alignment_summary.get("reliable_alignment", False)

    if usable_count > 1 and not reliable_alignment:
        template["warnings"].append(
            f"Spatial registration across usable images is unreliable: {alignment_summary.get('reason', '')}"
        )

    # Step 3: Feature Correspondence & Recurring Structures
    recurring_structures = _extract_recurring_structures(
        independent_records=independent_records,
        usable_count=usable_count,
        reliable_alignment=reliable_alignment,
        min_recurrence_ratio=min_recurrence_ratio,
    )
    template["recurring_graph_structures"] = recurring_structures

    # Step 4: Robust Consensus Statistics
    consensus_features: Dict[str, Dict[str, Any]] = {}

    # Collect all available descriptors across usable images
    all_descriptor_keys = list(STANDARD_DESCRIPTORS)
    for r in usable_records:
        g = r.get("graph_descriptors", {})
        if isinstance(g, dict):
            for k in g.keys():
                if k not in all_descriptor_keys:
                    all_descriptor_keys.append(k)

    for feat_name in all_descriptor_keys:
        values = []
        for r in usable_records:
            val = _extract_scalar(r, feat_name)
            if val is not None:
                values.append(val)

        if values:
            med, mad = robust_median_and_mad(values)
            mean_val = float(np.mean(values))
            stab = calculate_stability(values, med, mad)
            consensus_features[feat_name] = {
                "median": round(float(med), 3),
                "mad": round(float(mad), 3),
                "mean": round(float(mean_val), 3),
                "count": len(values),
                "stability": round(float(stab), 3),
            }

    template["consensus_features"] = consensus_features

    # Step 5: Overall Stability & Confidence Assessment
    topo_stabs = [
        consensus_features[k]["stability"]
        for k in TOPOLOGICAL_DESCRIPTOR_KEYS
        if k in consensus_features
    ]

    if topo_stabs:
        overall_stability = round(float(np.mean(topo_stabs)), 3)
    elif consensus_features:
        overall_stability = round(
            float(np.mean([v["stability"] for v in consensus_features.values()])), 3
        )
    else:
        overall_stability = 0.0

    # Determine confidence level
    if independent_view_count == 1:
        confidence_level = "low"
    elif overall_stability < 0.65:
        confidence_level = "low"
    elif (
        independent_view_count >= 3
        and usable_ratio >= 0.6
        and overall_stability >= 0.80
    ):
        confidence_level = "high"
    elif independent_view_count >= 2 and overall_stability >= 0.65:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    template["feature_stability"] = {
        "overall_stability_score": overall_stability,
        "confidence_level": confidence_level,
    }

    if overall_stability < 0.65 and usable_count > 1:
        template["warnings"].append(
            f"Overall topological stability is low ({overall_stability:.3f} < 0.65). "
            "Descriptors show high dispersion across views."
        )

    template["usable_count"] = usable_count
    template["submitted_count"] = submitted_count
    template["independent_view_count"] = independent_view_count
    template["confidence_level"] = confidence_level
    template["overall_stability_score"] = overall_stability

    return template
