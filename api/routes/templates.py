"""
Multi-Image Biometric Reference Template Routes
===============================================
Endpoints for consolidating multi-image batches into canonical topological
biometric reference templates, cross-template comparison, and CSV/JSON export.
"""

import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..helpers import BASE_DIR, OUTPUT_DIR, TEMPLATES_DIR, TEST_SAMPLES_DIR
from ml.batch_processor import process_image_batch
from ml.groove_graph import extract_skeleton_graph, visualize_groove_graph
from ml.template_consensus import build_consensus_template
from ml.template_matcher import (
    save_reference_template,
    load_reference_template,
    list_saved_templates,
    compare_two_templates,
)
from ml.template_exporter import (
    export_all_template_reports,
    export_consensus_summary_csv,
    export_per_image_measurements_csv,
    export_template_json,
)
from ml.template_schema import make_empty_template

router = APIRouter()


class BuildTemplateBatchRequest(BaseModel):
    image_paths: List[str]
    template_id: Optional[str] = None
    duplicate_threshold: float = 0.92
    method: str = "blackhat"
    scales: List[int] = Field(default_factory=lambda: [9, 15, 21])
    threshold_method: str = "otsu"
    min_component_size: int = 15


@router.post("/api/template/build-from-batch")
async def build_template_from_batch(req: BuildTemplateBatchRequest):
    """
    Ingests up to 5 images, assesses quality, extracts topological groove graphs,
    and consolidates them into a robust reference template.
    """
    if not req.image_paths:
        raise HTTPException(status_code=400, detail="At least 1 image path required.")

    # Resolve paths
    resolved_paths = []
    for p_str in req.image_paths[:5]:
        p = Path(p_str)
        if not p.is_absolute():
            p = BASE_DIR / p_str
        if not p.exists():
            cand1 = TEST_SAMPLES_DIR / Path(p_str).name
            cand2 = OUTPUT_DIR / p_str
            if cand1.exists():
                p = cand1
            elif cand2.exists():
                p = cand2
            else:
                raise HTTPException(status_code=404, detail=f"Image not found: {p_str}")
        resolved_paths.append(str(p))

    tmpl_id = req.template_id or f"tmpl_{int(time.time() * 1000)}"
    batch_out_dir = TEMPLATES_DIR / tmpl_id
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    def _execute_batch():
        batch_res = process_image_batch(
            images=resolved_paths,
            output_dir=batch_out_dir,
            groove_params={
                "method": req.method,
                "scales": req.scales,
                "threshold_method": req.threshold_method,
                "min_component_size": req.min_component_size,
            },
            duplicate_threshold=req.duplicate_threshold,
            batch_id=tmpl_id,
        )

        records = batch_res["records"]
        for idx, r in enumerate(records):
            if r.get("is_usable") and r.get("groove_skeleton") is not None:
                g_res = extract_skeleton_graph(r["groove_skeleton"], mask=r.get("clean_mask"))
                r["descriptors"] = g_res["descriptors"]
                r["branches"] = g_res["branches"]
                r["centroid"] = g_res["spatial_centroid"]

                p_overlay = batch_out_dir / f"img_{idx:02d}_graph_overlay.png"
                overlay_img = visualize_groove_graph(r["masked_image"], g_res)
                cv2.imwrite(str(p_overlay), overlay_img)
                if "saved_paths" not in r:
                    r["saved_paths"] = {}
                r["saved_paths"]["graph_overlay"] = str(p_overlay)
            else:
                r["descriptors"] = {}
                r["branches"] = []

        template = build_consensus_template(records, template_id=tmpl_id)
        save_reference_template(template, batch_out_dir / f"{tmpl_id}.json")
        export_all_template_reports(template, batch_out_dir)

        rel_base = f"/output/templates/{tmpl_id}"
        template["artifacts"] = {
            "template_json_url": f"{rel_base}/{tmpl_id}.json",
            "measurements_csv_url": f"{rel_base}/per_image_measurements.csv",
            "consensus_csv_url": f"{rel_base}/consensus_summary.csv",
        }
        for idx, s in enumerate(template.get("source_images", [])):
            s["image_url"] = f"{rel_base}/img_{idx:02d}_native_crop.png"
            s["skeleton_url"] = f"{rel_base}/img_{idx:02d}_groove_skeleton.png"
            s["graph_overlay_url"] = f"{rel_base}/img_{idx:02d}_graph_overlay.png"

        return template

    loop = asyncio.get_event_loop()
    try:
        template = await loop.run_in_executor(None, _execute_batch)
        return template
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch template building failed: {str(e)}")


@router.get("/api/template/sample-batch")
async def get_sample_template():
    """Returns pre-computed or generated 5-image reference template for instant UI display."""
    sample_json = TEMPLATES_DIR / "sample_5image_consensus" / "sample_5image_consensus.json"
    if sample_json.exists():
        try:
            return load_reference_template(sample_json)
        except Exception:
            pass

    sample_images = list(TEST_SAMPLES_DIR.glob("*.jpg"))
    if not sample_images:
        sample_images = list(OUTPUT_DIR.glob("**/*crop*.jpg"))

    if not sample_images:
        return make_empty_template(template_id="sample_template_default", submitted_count=5)

    img_paths = [str(p) for p in (sample_images * 5)[:5]]
    req = BuildTemplateBatchRequest(image_paths=img_paths, template_id="sample_5image_consensus")
    return await build_template_from_batch(req)


@router.post("/api/template/save")
async def save_template_endpoint(data: Dict[str, Any] = Body(...)):
    """Saves reference template JSON to storage."""
    try:
        t = data.get("template") or data
        t_id = t.get("template_id") or f"tmpl_{int(time.time() * 1000)}"
        out_p = TEMPLATES_DIR / f"{t_id}.json"
        path_str = save_reference_template(t, out_p)
        return {"success": True, "template_id": t_id, "file_path": path_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save template: {str(e)}")


@router.get("/api/template/list")
async def list_templates_endpoint():
    """Lists saved reference templates."""
    try:
        templates = list_saved_templates(TEMPLATES_DIR)
        return {"success": True, "templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


class CompareTemplatesRequest(BaseModel):
    template_a: Any
    template_b: Any
    decision_threshold: Optional[float] = 0.72


@router.post("/api/template/compare")
async def compare_templates_endpoint(req: CompareTemplatesRequest):
    """Compares two reference templates and returns statistical biometric match verdict."""
    try:
        if isinstance(req.template_a, str):
            t_a = load_reference_template(req.template_a, storage_dir=TEMPLATES_DIR)
        else:
            t_a = req.template_a

        if isinstance(req.template_b, str):
            t_b = load_reference_template(req.template_b, storage_dir=TEMPLATES_DIR)
        else:
            t_b = req.template_b

        res = compare_two_templates(
            t_a,
            t_b,
            decision_threshold=req.decision_threshold or 0.72,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Template comparison failed: {str(e)}")


@router.get("/api/template/export/json/{template_id}")
async def export_template_json_endpoint(template_id: str):
    """Downloads canonical JSON reference template."""
    p = TEMPLATES_DIR / f"{template_id}.json"
    if not p.exists():
        p = TEMPLATES_DIR / template_id / f"{template_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return FileResponse(p, media_type="application/json", filename=f"{template_id}.json")


@router.get("/api/template/export/csv/{export_type}/{template_id}")
async def export_template_csv_endpoint(export_type: str, template_id: str):
    """Downloads per-image measurements or consensus summary CSV."""
    dir_p = TEMPLATES_DIR / template_id
    if export_type == "measurements":
        fname = "per_image_measurements.csv"
    elif export_type == "consensus":
        fname = "consensus_summary.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid export_type. Choose 'measurements' or 'consensus'")

    file_p = dir_p / fname
    if not file_p.exists():
        raise HTTPException(status_code=404, detail=f"CSV file {fname} not found for template {template_id}")
    return FileResponse(file_p, media_type="text/csv", filename=f"{template_id}_{fname}")
