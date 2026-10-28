"""
Biometric Reference Template Exporter
=====================================
Exports consolidated multi-image reference templates to structured JSON
and tabular CSV files (per-image measurements and consensus feature summaries).
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def export_template_json(template: Dict[str, Any], output_path: str) -> str:
    """Exports template as formatted JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    return str(p)


def export_per_image_measurements_csv(template: Dict[str, Any], output_path: str) -> str:
    """
    Exports tabular CSV of per-image measurements:
    image_index, source_id, is_usable, is_duplicate, quality_score, sharpness,
    contrast, valid_area_pct, skeleton_pixels, endpoints, junctions, components,
    mean_branch_len, cycles, euler_char, rejection_reasons.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_index",
        "source_id",
        "is_usable",
        "is_duplicate",
        "quality_score",
        "sharpness",
        "contrast",
        "valid_area_pct",
        "skeleton_pixels",
        "endpoint_count",
        "junction_count",
        "connected_component_count",
        "mean_branch_length",
        "cycle_count",
        "euler_characteristic",
        "rejection_reasons",
    ]

    rows = []
    # Match source images with features
    features_by_idx = {
        f.get("image_index"): f.get("graph_descriptors", {})
        for f in template.get("per_image_features", [])
    }

    for src in template.get("source_images", []):
        idx = src.get("image_index")
        q = src.get("quality", {})
        g = features_by_idx.get(idx, {})
        reasons = "; ".join(src.get("rejection_reasons", []))

        row = {
            "image_index": idx,
            "source_id": src.get("source_id", f"img_{idx}"),
            "is_usable": src.get("is_usable", False),
            "is_duplicate": src.get("is_duplicate", False),
            "quality_score": q.get("quality_score", 0.0),
            "sharpness": q.get("sharpness", 0.0),
            "contrast": q.get("contrast", 0.0),
            "valid_area_pct": q.get("valid_area_pct", 0.0),
            "skeleton_pixels": g.get("skeleton_pixel_count", 0),
            "endpoint_count": g.get("endpoint_count", 0),
            "junction_count": g.get("junction_count", 0),
            "connected_component_count": g.get("connected_component_count", 0),
            "mean_branch_length": round(g.get("mean_branch_length", 0.0), 2),
            "cycle_count": g.get("cycle_count", 0),
            "euler_characteristic": g.get("euler_characteristic", 0),
            "rejection_reasons": reasons,
        }
        rows.append(row)

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(p)


def export_consensus_summary_csv(template: Dict[str, Any], output_path: str) -> str:
    """
    Exports tabular CSV summarizing consensus features:
    feature_name, median, mad, mean, observation_count, stability_score.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "feature_name",
        "median",
        "mad",
        "mean",
        "observation_count",
        "stability_score",
    ]

    rows = []
    consensus = template.get("consensus_features", {})
    for feat_name, stat in consensus.items():
        if isinstance(stat, dict):
            rows.append({
                "feature_name": feat_name,
                "median": round(stat.get("median", 0.0), 3),
                "mad": round(stat.get("mad", 0.0), 3),
                "mean": round(stat.get("mean", 0.0), 3),
                "observation_count": stat.get("count", 0),
                "stability_score": round(stat.get("stability", 0.0), 3),
            })

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(p)


def export_all_template_reports(template: Dict[str, Any], export_dir: str) -> Dict[str, str]:
    """Generates all exports (JSON, per-image CSV, consensus CSV) into export_dir."""
    exp_p = Path(export_dir)
    exp_p.mkdir(parents=True, exist_ok=True)

    t_id = template.get("template_id", "muzzle_template")
    json_path = exp_p / f"{t_id}.json"
    per_img_csv = exp_p / f"{t_id}_per_image_measurements.csv"
    consensus_csv = exp_p / f"{t_id}_consensus_summary.csv"

    return {
        "json": export_template_json(template, str(json_path)),
        "per_image_csv": export_per_image_measurements_csv(template, str(per_img_csv)),
        "consensus_csv": export_consensus_summary_csv(template, str(consensus_csv)),
    }
