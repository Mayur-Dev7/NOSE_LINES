"""
Reference Template Storage and Statistical Two-Template Matching Module
========================================================================
Strictly scoped to /home/adis_mayur/F_CODES/nose_lines.

Implements:
1. Reference template JSON serialization and persistent storage.
2. Structured template loading with schema version validation.
3. Saved template index and summary extraction.
4. Two-template statistical biometric comparison:
   - Global descriptor similarity (consensus medians normalized by pooled MAD).
   - Recurring graph structure similarity (spatial & angular bipartite matching).
   - Composite biometric fusion (60% global, 40% recurring structure).
   - Verdict determination and forensic disclaimers.
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from template_schema import SCHEMA_VERSION, robust_median_and_mad, calculate_stability, make_empty_template

logger = logging.getLogger("template_matcher")

DEFAULT_TEMPLATES_DIR = PROJECT_ROOT / "output" / "templates"

DEFAULT_DECISION_THRESHOLD = 0.72

DEFAULT_GLOBAL_WEIGHT = 0.60
DEFAULT_STRUCTURE_WEIGHT = 0.40

MAX_SPATIAL_DISTANCE = 0.15
MAX_ORIENTATION_DIFF_DEG = 35.0

KEY_GLOBAL_FEATURES = [
    "skeleton_pixel_count",
    "skeleton_density",
    "endpoint_count",
    "junction_count",
    "mean_branch_length",
    "cycle_count",
    "euler_characteristic",
    "curvature_mean",
]

DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "skeleton_pixel_count": 0.15,
    "skeleton_density": 0.15,
    "endpoint_count": 0.15,
    "junction_count": 0.15,
    "mean_branch_length": 0.15,
    "cycle_count": 0.10,
    "euler_characteristic": 0.10,
    "curvature_mean": 0.05,
}

FORENSIC_DISCLAIMER = (
    "Match score indicates dermatoglyphic pattern agreement under current imaging conditions. "
    "It does not represent an uncalibrated forensic certainty unless validated against large-scale ground truth datasets."
)


def save_reference_template(
    template: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Saves a reference template dictionary as an indented JSON file in output/templates/{template_id}.json
    (or in output_dir/{template_id}.json if output_dir is specified).

    Ensures the target directory exists.
    Returns the resolved path string to the saved file.
    """
    template_id = template.get("template_id")
    if not template_id:
        template_id = f"template_{int(time.time() * 1000)}"
        template["template_id"] = template_id

    if output_dir is not None:
        p = Path(output_dir)
        if p.suffix == ".json":
            target_dir = p.parent
            file_path = p
        else:
            target_dir = p
            file_path = target_dir / f"{template_id}.json"
    else:
        target_dir = DEFAULT_TEMPLATES_DIR
        file_path = target_dir / f"{template_id}.json"

    target_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    return str(file_path.resolve())


def load_reference_template(
    template_path_or_id: Union[str, Path],
    templates_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Loads and parses a reference template JSON file.
    Validates that the schema version matches SCHEMA_VERSION (major version match).

    Accepts a direct file path (str or Path) or a template ID to look up in templates_dir.
    """
    base_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    p = Path(template_path_or_id)

    if not p.is_file():
        # Check in base_dir
        c1 = base_dir / f"{template_path_or_id}.json"
        if c1.is_file():
            p = c1
        else:
            c2 = base_dir / str(template_path_or_id)
            if c2.is_file():
                p = c2
            else:
                c3 = Path(f"{template_path_or_id}.json")
                if c3.is_file():
                    p = c3
                else:
                    raise FileNotFoundError(f"Reference template not found: {template_path_or_id}")

    with open(p, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt template JSON at {p}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid template format at {p}: root JSON element must be an object")

    schema_ver = data.get("schema_version")
    if not schema_ver:
        raise ValueError(f"Invalid template at {p}: missing 'schema_version'")

    actual_major = str(schema_ver).strip().split(".")[0]
    expected_major = SCHEMA_VERSION.strip().split(".")[0]
    if actual_major != expected_major:
        raise ValueError(
            f"Incompatible schema_version '{schema_ver}' in {p} (expected major version '{expected_major}')"
        )

    return data


def list_saved_templates(
    templates_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of metadata summaries for all stored templates in templates_dir:
    - id
    - date
    - submitted_count
    - usable_count
    - confidence_level
    - overall_stability
    """
    base_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    if not base_dir.is_dir():
        return []

    summaries: List[Dict[str, Any]] = []
    seen_ids = set()
    found_files = set(base_dir.glob("*.json")) | set(base_dir.glob("*/*.json"))
    json_files = sorted(found_files, key=lambda f: f.stat().st_mtime, reverse=True)

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "consensus_features" not in data:
                continue

            t_id = data.get("template_id") or jf.stem
            if t_id in seen_ids:
                continue
            seen_ids.add(t_id)

            batch_sum = data.get("batch_summary", {})
            feat_stab = data.get("feature_stability", {})
            date_val = data.get("created_at", "")

            summary = {
                "id": t_id,
                "template_id": t_id,
                "date": date_val,
                "created_at": date_val,
                "submitted_count": int(batch_sum.get("submitted_count", 0)),
                "usable_count": int(batch_sum.get("usable_count", 0)),
                "confidence_level": str(feat_stab.get("confidence_level", "unknown")),
                "overall_stability": float(feat_stab.get("overall_stability_score", 0.0)),
                "file_path": str(jf.resolve()),
            }
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"Failed to read template summary from {jf}: {e}")
            continue

    return summaries


def _extract_feature_stat(feat_val: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts (median, mad) from feature representation.
    Supports dictionary formats, numeric scalars, and handles missing/malformed values gracefully.
    """
    if feat_val is None:
        return None, None
    if isinstance(feat_val, (int, float)):
        return float(feat_val), 0.0
    if isinstance(feat_val, dict):
        med = feat_val.get("median")
        if med is None:
            med = feat_val.get("consensus_median")
        if med is None:
            med = feat_val.get("mean")
        if med is None:
            med = feat_val.get("value")

        mad = feat_val.get("mad")
        if mad is None:
            mad = feat_val.get("consensus_mad")
        if mad is None:
            mad = 0.0

        if med is not None:
            try:
                return float(med), float(mad)
            except (ValueError, TypeError):
                return None, None
    return None, None


def compute_global_descriptor_similarity(
    consensus_a: Dict[str, Any],
    consensus_b: Dict[str, Any],
    feature_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compares consensus medians for key features:
    skeleton_pixel_count, skeleton_density, endpoint_count, junction_count,
    mean_branch_length, cycle_count, euler_characteristic, curvature_mean.

    Normalizes difference by pooled MAD:
        z_i = |med_A - med_B| / (mad_A + mad_B + 1e-4)
    Maps to similarity in [0, 1]:
        s_i = exp(-0.5 * z_i)
    Global similarity = weighted average of s_i.
    """
    weights = feature_weights or DEFAULT_FEATURE_WEIGHTS

    total_weight = 0.0
    weighted_sim_sum = 0.0
    breakdown: Dict[str, Any] = {}

    for feat_name in KEY_GLOBAL_FEATURES:
        w_i = weights.get(feat_name, 1.0 / len(KEY_GLOBAL_FEATURES))
        val_a = consensus_a.get(feat_name)
        val_b = consensus_b.get(feat_name)

        med_a, mad_a = _extract_feature_stat(val_a)
        med_b, mad_b = _extract_feature_stat(val_b)

        if med_a is None and med_b is None:
            breakdown[feat_name] = {
                "median_a": None,
                "mad_a": None,
                "median_b": None,
                "mad_b": None,
                "z_score": None,
                "similarity": None,
                "weight": w_i,
                "status": "missing_both",
            }
            continue

        if med_a is None or med_b is None:
            # Asymmetric: present in one template, absent in the other
            s_i = 0.0
            z_i = 100.0
            total_weight += w_i
            weighted_sim_sum += w_i * s_i

            breakdown[feat_name] = {
                "median_a": round(med_a, 4) if med_a is not None else None,
                "mad_a": round(mad_a, 4) if mad_a is not None else None,
                "median_b": round(med_b, 4) if med_b is not None else None,
                "mad_b": round(mad_b, 4) if mad_b is not None else None,
                "z_score": z_i,
                "similarity": s_i,
                "weight": w_i,
                "status": "missing_one",
            }
            continue

        # Both present: compute normalized difference using pooled MAD
        pooled_mad = abs(mad_a) + abs(mad_b) + 1e-4
        z_i = abs(med_a - med_b) / pooled_mad
        z_clamped = min(z_i, 100.0)
        s_i = float(math.exp(-0.5 * z_clamped))

        total_weight += w_i
        weighted_sim_sum += w_i * s_i

        breakdown[feat_name] = {
            "median_a": round(med_a, 4),
            "mad_a": round(mad_a, 4),
            "median_b": round(med_b, 4),
            "mad_b": round(mad_b, 4),
            "z_score": round(float(z_i), 4),
            "similarity": round(float(s_i), 4),
            "weight": w_i,
            "status": "evaluated",
        }

    if total_weight > 0.0:
        global_similarity = weighted_sim_sum / total_weight
    else:
        global_similarity = 0.0

    return float(np.clip(global_similarity, 0.0, 1.0)), breakdown


def _parse_recurring_branch(item: Any) -> Optional[Tuple[float, float, float]]:
    """
    Parses a recurring branch structure element into (x, y, angle_deg).
    Supports:
    - Dict with 'centroid', 'center', 'midpoint', or 'position' [x, y]
    - Dict with 'x', 'y'
    - Dict with 'start', 'end' coordinates
    - Orientation keys: 'orientation', 'angle', 'theta', 'direction'
    - List or tuple [x, y, angle]
    """
    if item is None:
        return None

    x: Optional[float] = None
    y: Optional[float] = None
    angle: Optional[float] = None

    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
                angle = float(item[2]) if len(item) >= 3 else 0.0
            except (ValueError, TypeError):
                return None
    elif isinstance(item, dict):
        for pos_key in ("centroid", "center", "midpoint", "position"):
            if pos_key in item and isinstance(item[pos_key], (list, tuple)) and len(item[pos_key]) >= 2:
                try:
                    x = float(item[pos_key][0])
                    y = float(item[pos_key][1])
                    break
                except (ValueError, TypeError):
                    pass

        if x is None and "x" in item and "y" in item:
            try:
                x = float(item["x"])
                y = float(item["y"])
            except (ValueError, TypeError):
                pass

        if x is None and "start" in item and "end" in item:
            try:
                s, e = item["start"], item["end"]
                x = (float(s[0]) + float(e[0])) / 2.0
                y = (float(s[1]) + float(e[1])) / 2.0
                if angle is None:
                    dx = float(e[0]) - float(s[0])
                    dy = float(e[1]) - float(s[1])
                    angle = float(np.degrees(np.arctan2(dy, dx)))
            except (ValueError, TypeError, IndexError):
                pass

        for ang_key in ("orientation", "angle", "theta", "direction", "orientation_deg"):
            if ang_key in item:
                try:
                    angle = float(item[ang_key])
                    break
                except (ValueError, TypeError):
                    pass

    if x is not None and y is not None:
        if angle is None:
            angle = 0.0
        return (x, y, angle)

    return None


def _angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """
    Computes minimum angular difference in degrees [0, 90] for undirected ridge lines,
    accounting for 180-degree line symmetry and 360-degree modular wrap-around.
    """
    diff_360 = abs(angle_a - angle_b) % 360.0
    circular = min(diff_360, 360.0 - diff_360)

    diff_180 = abs(angle_a - angle_b) % 180.0
    undirected = min(diff_180, 180.0 - diff_180)

    return min(circular, undirected)


def compute_recurring_structure_similarity(
    structures_a: List[Any],
    structures_b: List[Any],
    max_spatial_distance: float = MAX_SPATIAL_DISTANCE,
    max_angle_diff: float = MAX_ORIENTATION_DIFF_DEG,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compares recurring graph branches from Template A and Template B.
    Matches branches within spatial distance <= 0.15 and orientation diff <= 35 deg.
    Structure similarity = 2 * |M| / max(1, |A_recurring| + |B_recurring|)
    """
    branches_a: List[Tuple[float, float, float]] = []
    for item in (structures_a or []):
        parsed = _parse_recurring_branch(item)
        if parsed is not None:
            branches_a.append(parsed)

    branches_b: List[Tuple[float, float, float]] = []
    for item in (structures_b or []):
        parsed = _parse_recurring_branch(item)
        if parsed is not None:
            branches_b.append(parsed)

    n_a = len(branches_a)
    n_b = len(branches_b)

    if n_a == 0 or n_b == 0:
        similarity = 0.0
        breakdown = {
            "template_a_count": n_a,
            "template_b_count": n_b,
            "matched_count": 0,
            "matches": [],
            "max_spatial_distance": max_spatial_distance,
            "max_angle_diff": max_angle_diff,
        }
        return similarity, breakdown

    # Find all mutually compatible candidate pairs
    candidates = []
    for i, (xa, ya, th_a) in enumerate(branches_a):
        for j, (xb, yb, th_b) in enumerate(branches_b):
            dist = math.hypot(xa - xb, ya - yb)
            if dist <= max_spatial_distance:
                angle_diff = _angle_difference_deg(th_a, th_b)
                if angle_diff <= max_angle_diff:
                    cost = dist + (angle_diff / 360.0)
                    candidates.append((cost, dist, angle_diff, i, j))

    # Greedy bipartite matching sorted by ascending cost
    candidates.sort(key=lambda c: c[0])

    matched_a = set()
    matched_b = set()
    matches = []

    for cost, dist, angle_diff, i, j in candidates:
        if i not in matched_a and j not in matched_b:
            matched_a.add(i)
            matched_b.add(j)
            matches.append({
                "a_index": i,
                "b_index": j,
                "spatial_distance": round(float(dist), 4),
                "orientation_diff_deg": round(float(angle_diff), 2),
            })

    m_count = len(matches)
    similarity = (2.0 * m_count) / float(max(1, n_a + n_b))
    similarity = float(np.clip(similarity, 0.0, 1.0))

    breakdown = {
        "template_a_count": n_a,
        "template_b_count": n_b,
        "matched_count": m_count,
        "matches": matches,
        "max_spatial_distance": max_spatial_distance,
        "max_angle_diff": max_angle_diff,
    }

    return similarity, breakdown


def compare_two_templates(
    template_a: Dict[str, Any],
    template_b: Dict[str, Any],
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    global_weight: float = DEFAULT_GLOBAL_WEIGHT,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
) -> Dict[str, Any]:
    """
    Compares two reference templates:
    1. Global Descriptor Similarity:
       - Compares consensus medians for key features:
         skeleton_pixel_count, skeleton_density, endpoint_count, junction_count,
         mean_branch_length, cycle_count, euler_characteristic, curvature_mean.
       - Normalizes difference by pooled MAD:
         z_i = |med_A - med_B| / (mad_A + mad_B + 1e-4)
       - Maps to similarity in [0, 1]:
         s_i = exp(-0.5 * z_i)
       - Global similarity = weighted average of s_i.
    2. Recurring Structure Similarity:
       - Compares recurring graph branches from Template A and Template B.
       - Matches branches within spatial distance <= 0.15 and orientation diff <= 35 deg.
       - Structure similarity = 2 * |M| / max(1, |A_recurring| + |B_recurring|).
    3. Composite Biometric Score:
       - Weighted fusion of global descriptor similarity (60%) and recurring structure similarity (40%).
    4. Verdict & Disclaimers:
       - is_match: boolean (composite_score >= decision_threshold)
       - verdict: "MATCH (Same Cattle Candidate)" or "DISTINCT (Different Animals)"
       - disclaimer: "Match score indicates dermatoglyphic pattern agreement under current imaging conditions. It does not represent an uncalibrated forensic certainty unless validated against large-scale ground truth datasets."
       - Returns complete comparison breakdown dictionary.
    """
    if not isinstance(template_a, dict) or not isinstance(template_b, dict):
        raise ValueError("Both template_a and template_b must be dictionaries")

    consensus_a = template_a.get("consensus_features") or {}
    consensus_b = template_b.get("consensus_features") or {}

    structures_a = template_a.get("recurring_graph_structures") or []
    structures_b = template_b.get("recurring_graph_structures") or []

    # 1. Global Descriptor Similarity
    global_sim, feat_breakdown = compute_global_descriptor_similarity(
        consensus_a, consensus_b
    )

    # 2. Recurring Structure Similarity
    struct_sim, struct_breakdown = compute_recurring_structure_similarity(
        structures_a, structures_b
    )

    # 3. Composite Biometric Score
    n_struct_a = struct_breakdown["template_a_count"]
    n_struct_b = struct_breakdown["template_b_count"]

    if n_struct_a == 0 and n_struct_b == 0:
        # Neither template provides recurring graph structures
        composite_score = global_sim
        effective_global_w = 1.0
        effective_struct_w = 0.0
    else:
        composite_score = (global_weight * global_sim) + (structure_weight * struct_sim)
        effective_global_w = global_weight
        effective_struct_w = structure_weight

    composite_score = float(np.clip(composite_score, 0.0, 1.0))

    # 4. Verdict & Disclaimers
    is_match = bool(composite_score >= decision_threshold)
    verdict = "MATCH (Same Cattle Candidate)" if is_match else "DISTINCT (Different Animals)"

    template_a_id = template_a.get("template_id", "template_a")
    template_b_id = template_b.get("template_id", "template_b")

    return {
        "is_match": is_match,
        "verdict": verdict,
        "composite_score": round(composite_score, 4),
        "decision_threshold": float(decision_threshold),
        "global_descriptor_similarity": round(global_sim, 4),
        "recurring_structure_similarity": round(struct_sim, 4),
        "fusion_weights": {
            "global_weight": effective_global_w,
            "structure_weight": effective_struct_w,
        },
        "feature_breakdown": feat_breakdown,
        "recurring_structure_breakdown": struct_breakdown,
        "template_a_id": template_a_id,
        "template_b_id": template_b_id,
        "disclaimer": FORENSIC_DISCLAIMER,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }
