"""
Deep Learning Biometric Embeddings Routes
=========================================
Endpoints for 1-to-N multi-cattle biometric matching, gallery presets management,
and single-video 512-D ArcFace / Spatial GeM embedding vector export.
"""

import json
import time
import asyncio
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from ..log_handler import logger
from ..helpers import (
    BASE_DIR,
    OUTPUT_DIR,
    resolve_video_path,
    load_gallery_presets_data,
    _extract_embedding_from_summary,
    _EXTRACTION_LOCK,
)
from ml.video_groove_extractor import extract_five_frames_groove_features

router = APIRouter()


@router.get("/api/embedding/gallery-presets")
async def get_embedding_gallery_presets_endpoint(reload: bool = Query(False)):
    """
    Returns a pre-packaged gallery of registered cattle embeddings from
    existing summary files in output/video_five_frames/ (e.g. muzzleVideo_frangi_adaptive
    as Cattle A Profile 1, muzzleVideo2_frangi_adaptive as Cattle A Profile 2,
    muzzleVideo3_frangi_adaptive as Cattle B) or compute them on demand.
    """
    gallery = load_gallery_presets_data(force_reload=reload)

    # Check if samples defined in config/gallery_presets.json
    presets_json_path = BASE_DIR / "config" / "gallery_presets.json"
    sample_a = None
    sample_b = None
    if presets_json_path.exists():
        try:
            raw_data = json.loads(presets_json_path.read_text(encoding="utf-8"))
            sample_a = raw_data.get("sample_cow_a")
            sample_b = raw_data.get("sample_cow_b")
        except Exception:
            pass

    if not sample_a and gallery:
        for cand in gallery:
            if "cattle_001_a" in cand["id"] or "profile 1" in cand["name"].lower():
                sample_a = {"id": cand["id"], "name": cand["name"], "embedding": cand["embedding"]}
                break
        if not sample_a:
            sample_a = {"id": gallery[0]["id"], "name": gallery[0]["name"], "embedding": gallery[0]["embedding"]}

    if not sample_b and len(gallery) > 1:
        for cand in gallery:
            if "cattle_002" in cand["id"] or "cattle b" in cand["name"].lower():
                sample_b = {"id": cand["id"], "name": cand["name"], "embedding": cand["embedding"]}
                break
        if not sample_b:
            sample_b = {"id": gallery[1]["id"], "name": gallery[1]["name"], "embedding": gallery[1]["embedding"]}

    return {
        "success": True,
        "gallery": gallery,
        "sample_cow_a": sample_a,
        "sample_cow_b": sample_b,
    }


class OneToNMatchRequest(BaseModel):
    query_embedding: List[float]
    gallery: Optional[List[Dict[str, Any]]] = None
    similarity_metric: Optional[str] = "cosine"  # "cosine" or "l2"
    threshold: Optional[float] = 88.0
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6


@router.post("/api/embedding/1-to-n-match")
async def one_to_n_match_endpoint(req: OneToNMatchRequest):
    """
    1-to-N Multi-Cattle Biometric Matching Endpoint:
    Matches a query 512-D embedding against a gallery matrix G (N, 512).
    Vectorized NumPy dot-product computation, Euclidean L2 distance,
    biometric thresholding verdicts, runner-up margin, and candidate ranking.
    """
    t_match_0 = time.perf_counter()
    if not req.query_embedding or len(req.query_embedding) == 0:
        raise HTTPException(status_code=400, detail="query_embedding cannot be empty.")

    # 1. Normalize query vector to unit length: q = q / max(1e-8, ||q||_2)
    q = np.array(req.query_embedding, dtype=np.float32)
    dim = int(len(q))
    q_norm = float(np.linalg.norm(q))
    q = q / max(1e-8, q_norm)

    gallery_candidates = req.gallery
    if gallery_candidates is None:
        gallery_candidates = load_gallery_presets_data()

    N = len(gallery_candidates)
    if N == 0:
        return {
            "success": True,
            "query_dim": dim,
            "total_candidates": 0,
            "top_match": None,
            "margin_to_runner_up": None,
            "ranked_results": [],
            "margin_ahead_pct": 0.0,
            "leaderboard": [],
            "total_gallery_candidates": 0,
            "verdict": "NO MATCH (Empty Gallery)",
            "verdict_is_match": False,
            "verdict_color": "rose",
        }

    # 2. Extract and validate candidate embeddings
    candidate_embeddings = []
    for idx, item in enumerate(gallery_candidates):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Gallery candidate at index {idx} must be a dictionary.")
        raw_emb = item.get("embedding")
        if raw_emb is None:
            raise HTTPException(status_code=400, detail=f"Candidate at index {idx} missing 'embedding' field.")
        if isinstance(raw_emb, dict):
            raw_emb = raw_emb.get("median_embedding") or raw_emb.get("mean_embedding")
            if not raw_emb:
                raise HTTPException(status_code=400, detail=f"Candidate at index {idx} embedding object has no median/mean vector.")
        if len(raw_emb) != dim:
            raise HTTPException(
                status_code=400,
                detail=f"Dimension mismatch at candidate {idx} ('{item.get('id', idx)}'): query has {dim} dimensions, candidate has {len(raw_emb)}.",
            )
        candidate_embeddings.append(raw_emb)

    # 3. Stack gallery embeddings into matrix G (N, 512), normalize each row to unit length
    G = np.array(candidate_embeddings, dtype=np.float32)
    row_norms = np.linalg.norm(G, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    G = G / row_norms

    # 4. Compute dot products in one vectorized NumPy call
    sims = np.dot(G, q)  # (N,)
    sim_pcts = np.clip(sims, 0.0, 1.0) * 100.0
    l2_dists = np.linalg.norm(G - q, axis=1)  # (N,)

    # 5. For each candidate, determine verdict
    threshold = float(req.threshold if req.threshold is not None else 88.0)
    ranked_results = []
    for i, item in enumerate(gallery_candidates):
        s_pct = float(sim_pcts[i])
        s_val = float(sims[i])
        dist = float(l2_dists[i])

        if s_pct >= 88.0:
            verdict = "MATCH (Confirmed Same Cattle)"
            badge = "match"
            color = "emerald"
        elif s_pct >= 78.0:
            verdict = "LIKELY MATCH"
            badge = "likely"
            color = "teal"
        elif s_pct >= 65.0:
            verdict = "INCONCLUSIVE"
            badge = "inconclusive"
            color = "amber"
        else:
            verdict = "DISTINCT (Different Animals)"
            badge = "distinct"
            color = "rose"

        meta_dict = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        cand_dict = {
            "id": str(item.get("id", f"candidate_{i}")),
            "name": str(item.get("name", f"Candidate {i+1}")),
            "similarity": round(s_val, 4),
            "similarity_pct": round(s_pct, 2),
            "l2_distance": round(dist, 4),
            "verdict": verdict,
            "badge": badge,
            "color": color,
            "verdict_color": color,
            "notes": item.get("notes"),
            "tag": item.get("tag") or meta_dict.get("tag") or "N/A",
            "breed": item.get("breed") or meta_dict.get("breed") or "Unknown",
            "registered_date": item.get("registered_date") or meta_dict.get("registered_date") or "",
            "meta": meta_dict,
            "is_match": bool(s_pct >= threshold),
        }
        ranked_results.append(cand_dict)

    # 6. Rank candidates descending by similarity score
    ranked_results.sort(key=lambda x: x["similarity_pct"], reverse=True)
    for rank_idx, r in enumerate(ranked_results, start=1):
        r["rank"] = rank_idx

    top_match = ranked_results[0] if ranked_results else None
    margin_to_runner_up = (
        round(float(ranked_results[0]["similarity_pct"] - ranked_results[1]["similarity_pct"]), 2)
        if len(ranked_results) > 1
        else None
    )

    is_confirmed = bool(top_match and top_match["similarity_pct"] >= threshold)
    return {
        "success": True,
        "query_dim": dim,
        "total_candidates": len(ranked_results),
        "top_match": top_match,
        "margin_to_runner_up": margin_to_runner_up,
        "ranked_results": ranked_results,
        # Backward compatibility and UI integration fields
        "margin_ahead_pct": margin_to_runner_up if margin_to_runner_up is not None else 0.0,
        "leaderboard": ranked_results,
        "total_gallery_candidates": len(ranked_results),
        "verdict": f"MATCH FOUND: {top_match['name']}" if is_confirmed else "NO MATCH (All Distinct Animals)",
        "verdict_is_match": is_confirmed,
        "verdict_color": "emerald" if is_confirmed else "rose",
        "threshold": threshold,
        "step_timings": {
            "step_7_consensus_ms": max(0.5, round((time.perf_counter() - t_match_0) * 1000, 1)),
            "total_ms": max(0.5, round((time.perf_counter() - t_match_0) * 1000, 1)),
        },
    }


# Backwards compatibility aliases
match_one_to_n_endpoint = one_to_n_match_endpoint
get_gallery_presets_endpoint = get_embedding_gallery_presets_endpoint


class SingleVideoEmbeddingExportRequest(BaseModel):
    video_path: str
    pool_method: Optional[str] = "median"  # 'median' or 'mean'
    method: Optional[str] = "frangi"
    threshold_method: Optional[str] = "adaptive"
    adaptive_block_size: Optional[int] = 15
    adaptive_c: Optional[int] = 2
    exclude_nostrils: Optional[bool] = True
    device: Optional[str] = "cpu"
    cpu_cores: Optional[int] = 6
    min_spur_length: Optional[int] = 8
    min_component_size: Optional[int] = 30
    dl_backbone: Optional[str] = "convnext_tiny"
    dl_input_mode: Optional[str] = "composite_3ch"


async def _handle_single_video_embedding_export(
    video_path: str,
    pool_method: str = "median",
    req_opts: Optional[SingleVideoEmbeddingExportRequest] = None,
) -> Dict[str, Any]:
    resolved_v = resolve_video_path(video_path)
    if not resolved_v or not resolved_v.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    chosen_pool = pool_method or "median"

    # Search existing summary.json in output/video_five_frames/
    vff_dir = OUTPUT_DIR / "video_five_frames"
    stem = resolved_v.stem
    cand_summaries = list(vff_dir.glob(f"*{stem}*/summary.json"))
    name_no_ts = "_".join(stem.split("_")[1:]) if "_" in stem else stem
    if name_no_ts and name_no_ts != stem:
        cand_summaries.extend(vff_dir.glob(f"*{name_no_ts}*/summary.json"))

    for c_path in sorted(cand_summaries, reverse=True):
        try:
            data = json.loads(c_path.read_text(encoding="utf-8"))
            dl = data.get("dl_embedding")
            emb = _extract_embedding_from_summary(data, pool_method=chosen_pool)
            if emb:
                return {
                    "success": True,
                    "video_name": resolved_v.name,
                    "video_path": str(resolved_v),
                    "embedding": emb,
                    "embedding_dim": len(emb),
                    "pool_method": chosen_pool,
                    "intra_video_consistency_pct": dl.get("intra_video_consistency_pct") if dl else None,
                    "cached": True,
                    "summary_path": str(c_path.relative_to(BASE_DIR)),
                    "dl_summary": {
                        "backbone": dl.get("backbone") if dl else "convnext_tiny",
                        "input_mode": dl.get("input_mode") if dl else "composite_3ch",
                        "weights_mode": dl.get("weights_mode") if dl else "pretrained_gem_zeroshot",
                        "avg_inference_ms": dl.get("avg_inference_ms") if dl else None,
                    } if dl else {},
                }
        except Exception as exc:
            logger.debug(f"Error reading summary {c_path}: {exc}")

    # Not cached: compute on demand
    fn = partial(
        extract_five_frames_groove_features,
        video_path=str(resolved_v),
        method=req_opts.method if req_opts else "frangi",
        threshold_method=req_opts.threshold_method if req_opts else "adaptive",
        adaptive_block_size=int(req_opts.adaptive_block_size or 15) if req_opts else 15,
        adaptive_c=int(req_opts.adaptive_c or 2) if req_opts else 2,
        exclude_nostrils=bool(req_opts.exclude_nostrils if (req_opts and req_opts.exclude_nostrils is not None) else True),
        device=req_opts.device if req_opts else "cpu",
        cpu_cores=int(req_opts.cpu_cores or 6) if req_opts else 6,
        min_spur_length=int(req_opts.min_spur_length or 8) if req_opts else 8,
        min_component_size=int(req_opts.min_component_size or 30) if req_opts else 30,
        dl_backbone=req_opts.dl_backbone if req_opts else "convnext_tiny",
        dl_input_mode=req_opts.dl_input_mode if req_opts else "composite_3ch",
        enable_dl_embedding=True,
    )
    async with _EXTRACTION_LOCK:
        res = await asyncio.to_thread(fn)

    emb = _extract_embedding_from_summary(res, pool_method=chosen_pool)
    if not emb:
        raise HTTPException(status_code=500, detail="Failed to extract biometric embedding from video.")

    dl = res.get("dl_embedding") or {}
    return {
        "success": True,
        "video_name": resolved_v.name,
        "video_path": str(resolved_v),
        "embedding": emb,
        "embedding_dim": len(emb),
        "pool_method": chosen_pool,
        "intra_video_consistency_pct": dl.get("intra_video_consistency_pct"),
        "cached": False,
        "dl_summary": {
            "backbone": dl.get("backbone"),
            "input_mode": dl.get("input_mode"),
            "weights_mode": dl.get("weights_mode"),
            "avg_inference_ms": dl.get("avg_inference_ms"),
        },
    }


@router.post("/api/embedding/export")
async def export_single_video_embedding_post(req: SingleVideoEmbeddingExportRequest):
    """
    POST /api/embedding/export:
    Exports single cattle video 512-D ArcFace/GeM Biometric Embedding vector.
    """
    return await _handle_single_video_embedding_export(
        video_path=req.video_path,
        pool_method=req.pool_method or "median",
        req_opts=req,
    )


@router.get("/api/embedding/export")
async def export_single_video_embedding_get(
    video_path: str = Query(..., description="Path or filename of video"),
    pool_method: Optional[str] = Query("median", description="'median' or 'mean'"),
):
    """
    GET /api/embedding/export:
    Exports single cattle video 512-D ArcFace/GeM Biometric Embedding vector.
    """
    return await _handle_single_video_embedding_export(
        video_path=video_path,
        pool_method=pool_method or "median",
    )
