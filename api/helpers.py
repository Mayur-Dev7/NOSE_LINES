"""
Shared Helpers, Caches, Constants, and Path Resolvers
=====================================================
Common utility functions, process path resolution, extraction cache,
and subprocess streaming used across the Cattle Muzzle AI API.
"""

import os
import sys
import glob
import json
import math
import time
import shutil
import asyncio
import functools
import subprocess
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Any, Dict, Union

import numpy as np
from .log_handler import logger
from ml.video_groove_extractor import extract_five_frames_groove_features

# Polyfill asyncio.to_thread for Python 3.8
if not hasattr(asyncio, "to_thread"):
    async def _asyncio_to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_event_loop()
        func_call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func_call)
    asyncio.to_thread = _asyncio_to_thread

# Base directory is the project root (parent of api/)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

WEIGHTS_PATH = Path("/home/adis_mayur/F_CODES/crop11dec.h5")
PYTHON_EXEC = BASE_DIR / ".venv" / "bin" / "python"
CROPPER_SCRIPT = BASE_DIR / "muzzle_cropper.py"
MATCHER_SCRIPT = BASE_DIR / "muzzle_matcher.py"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
TEST_SAMPLES_DIR = BASE_DIR / "test_samples"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
TEMPLATES_DIR = OUTPUT_DIR / "templates"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def parse_cpu_cores(cpu_cores: Optional[Any]) -> Optional[int]:
    """Parses arbitrary user-specified CPU core counts, validating integer >= 1."""
    if cpu_cores is None:
        return None
    val_str = str(cpu_cores).strip()
    if not val_str or val_str.lower() in ("null", "none", "auto", "all"):
        return None
    try:
        val = int(val_str)
        return val if val >= 1 else None
    except (ValueError, TypeError):
        return None


def stream_subprocess_output(cmd: List[str], env_extra: Optional[dict] = None):
    """Generator streaming stdout/stderr as SSE events."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    try:
        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("[JSON_EVENT]"):
                json_part = line_str[len("[JSON_EVENT]"):].strip()
                try:
                    payload = json.loads(json_part)
                    p_type = payload.get("type", "")

                    if p_type == "matcher_log":
                        yield f"data: {json.dumps({'event': 'log', 'line': payload.get('text', '')})}\n\n"
                    elif p_type == "two_video_result":
                        res_data = payload.get("result", {})
                        # Convert paths to URLs
                        for key in ("video_a", "video_b"):
                            sub = res_data.get(key, {})
                            for pkey, ukey in [
                                ("crop_path", "crop_url"),
                                ("annotated_path", "annotated_url"),
                                ("enhanced_path", "enhanced_url"),
                                ("groove_response_path", "groove_response_url"),
                                ("groove_mask_path", "groove_mask_url"),
                                ("groove_skeleton_path", "groove_skeleton_url"),
                                ("groove_overlay_path", "groove_overlay_url"),
                            ]:
                                if pkey in sub and sub[pkey]:
                                    try:
                                        sub[ukey] = "/output/" + str(Path(sub[pkey]).relative_to(OUTPUT_DIR))
                                    except Exception:
                                        if ukey not in sub or not sub[ukey]:
                                            sub[ukey] = f"/output/{Path(sub[pkey]).name}"
                        m = res_data.get("match", {})
                        if "visual_path" in m and m["visual_path"]:
                            try:
                                m["visual_url"] = "/output/" + str(Path(m["visual_path"]).relative_to(OUTPUT_DIR))
                            except Exception:
                                m["visual_url"] = m["visual_path"]
                        yield f"data: {json.dumps({'event': 'two_video_result', 'data': res_data})}\n\n"
                    else:
                        # Convert absolute paths to API-relative URLs for frontend
                        if "annotated_path" in payload and payload["annotated_path"]:
                            try:
                                payload["annotated_url"] = "/output/" + str(Path(payload["annotated_path"]).relative_to(OUTPUT_DIR))
                            except Exception:
                                pass
                        if "crops" in payload:
                            for c in payload["crops"]:
                                if "crop_path" in c and c["crop_path"]:
                                    try:
                                        c["crop_url"] = "/output/" + str(Path(c["crop_path"]).relative_to(OUTPUT_DIR))
                                    except Exception:
                                        pass
                        yield f"data: {json.dumps({'event': 'detection', 'data': payload})}\n\n"
                except Exception as ex:
                    yield f"data: {json.dumps({'event': 'log', 'line': line_str})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'log', 'line': line_str})}\n\n"

        proc.stdout.close()
        returncode = proc.wait()
        yield f"data: {json.dumps({'event': 'done', 'returncode': returncode})}\n\n"
    except GeneratorExit:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception as e:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"


def resolve_video_path(v_path: str) -> Path:
    p = Path(v_path)
    if p.exists():
        return p.resolve()
    cand_base = BASE_DIR / v_path
    if cand_base.exists():
        return cand_base.resolve()
    cand_root = Path("/home/adis_mayur/F_CODES") / v_path
    if cand_root.exists():
        return cand_root.resolve()
    cand_upload = UPLOADS_DIR / v_path
    if cand_upload.exists():
        return cand_upload.resolve()
    if not p.is_absolute():
        matches = sorted(UPLOADS_DIR.glob(f"*_{p.name}"))
        if matches:
            return matches[-1].resolve()
        matches_stem = sorted(UPLOADS_DIR.glob(f"*{p.stem}*.mp4"))
        if matches_stem:
            return matches_stem[-1].resolve()
    return p


def resolve_image_path(img_path: str) -> Optional[Path]:
    """
    Resolves an image or mask path across absolute paths, output directory,
    test_samples, uploads, and workspace directories.
    """
    if not img_path or not str(img_path).strip():
        return None
    p = Path(img_path)
    if p.is_file():
        return p.resolve()

    clean_p = str(img_path).strip().lstrip("/")
    # Check /output/... or output/...
    if clean_p.startswith("output/"):
        rel = clean_p[len("output/"):]
        cand = OUTPUT_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check /test_samples/... or test_samples/...
    if clean_p.startswith("test_samples/"):
        rel = clean_p[len("test_samples/"):]
        cand = TEST_SAMPLES_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check /uploads/... or uploads/...
    if clean_p.startswith("uploads/"):
        rel = clean_p[len("uploads/"):]
        cand = UPLOADS_DIR / rel
        if cand.is_file():
            return cand.resolve()

    # Check relative to OUTPUT_DIR
    cand = OUTPUT_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to BASE_DIR
    cand = BASE_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to TEST_SAMPLES_DIR
    cand = TEST_SAMPLES_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to UPLOADS_DIR
    cand = UPLOADS_DIR / clean_p
    if cand.is_file():
        return cand.resolve()

    # Check relative to /home/adis_mayur/F_CODES
    cand = Path("/home/adis_mayur/F_CODES") / clean_p
    if cand.is_file():
        return cand.resolve()

    return None


# Cache holds NumPy arrays only in backend memory. JSON responses are always plain data.
_EXTRACTION_LOCK = asyncio.Lock()
_EXTRACTION_CACHE = OrderedDict()
_EXTRACTION_CACHE_MAX = 8


def _sanitize_floats(obj):
    """Recursively replace NaN/Inf float values with 0.0 so JSON serialization never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return None  # Strip arrays
    return obj


def _json_safe_frame(frame):
    return _sanitize_floats({k: v for k, v in frame.items() if not isinstance(v, np.ndarray)})


def _spatial_cache_key(video_path, method, threshold_method, adaptive_block_size, adaptive_c,
                       exclude_nostrils, min_spur_length, min_component_size, device, cpu_cores,
                       dl_backbone="convnext_tiny", dl_input_mode="composite_3ch"):
    stat = Path(video_path).stat()
    signature = f"{Path(video_path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{method}:{threshold_method}:{adaptive_block_size}:{adaptive_c}:{exclude_nostrils}:{min_spur_length}:{min_component_size}:{device}:{cpu_cores}:{dl_backbone}:{dl_input_mode}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _put_extraction_cache(key, frames):
    _EXTRACTION_CACHE[key] = frames
    _EXTRACTION_CACHE.move_to_end(key)
    while len(_EXTRACTION_CACHE) > _EXTRACTION_CACHE_MAX:
        _EXTRACTION_CACHE.popitem(last=False)


def _get_extraction_cache(key):
    frames = _EXTRACTION_CACHE.get(key)
    if frames is not None:
        _EXTRACTION_CACHE.move_to_end(key)
    return frames


def _extract_video_for_spatial_match(video_path, req):
    """Synchronous extraction helper; caller controls the extraction lock."""
    return extract_five_frames_groove_features(
        video_path=str(video_path),
        method=req.filter_method or "frangi",
        threshold_method=req.threshold_method or "adaptive",
        adaptive_block_size=int(req.adaptive_block_size or 15),
        adaptive_c=int(req.adaptive_c or 2),
        exclude_nostrils=bool(req.exclude_nostrils if req.exclude_nostrils is not None else True),
        device=req.device or "cpu",
        cpu_cores=int(req.cpu_cores or 6),
        min_spur_length=int(req.min_spur_length or 8),
        min_component_size=int(req.min_component_size or 30),
        dl_backbone=getattr(req, "dl_backbone", "convnext_tiny") or "convnext_tiny",
        dl_input_mode=getattr(req, "dl_input_mode", "composite_3ch") or "composite_3ch",
        enable_dl_embedding=True,
    )


def _extract_embedding_from_summary(summary_data: Dict[str, Any], pool_method: str = "median") -> Optional[List[float]]:
    """Helper to extract an L2-normalized 512-D embedding list from summary data."""
    if not isinstance(summary_data, dict):
        return None
    dl = summary_data.get("dl_embedding")
    if not dl or not isinstance(dl, dict):
        return None

    emb = None
    if pool_method == "mean":
        emb = dl.get("mean_embedding") or dl.get("median_embedding")
    else:
        emb = dl.get("median_embedding") or dl.get("mean_embedding")

    if not emb and "per_frame_embeddings" in dl and dl["per_frame_embeddings"]:
        frames = dl["per_frame_embeddings"]
        if pool_method == "mean":
            emb = np.mean(frames, axis=0).tolist()
        else:
            emb = np.median(frames, axis=0).tolist()

    if emb:
        arr = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        return [round(float(x), 5) for x in arr]
    return None


GALLERY_PRESET_DEFINITIONS = [
    {
        "id": "cattle_001_a",
        "name": "Cattle A (Profile 1)",
        "folders": ["muzzleVideo_frangi_adaptive", "1793364335_muzzleVideo_frangi_adaptive", "muzzleVideo"],
        "video_path": "uploads/1793364335_muzzleVideo.mp4",
        "notes": "Registered Baseline",
        "tag": "US-94210-A",
        "breed": "Holstein Friesian",
        "registered_date": "2024-09-10",
        "meta": {"source_video": "muzzleVideo.mp4", "breed": "Holstein Friesian", "registration_id": "RF-9021", "animal_id": "cattle_001"},
    },
    {
        "id": "cattle_001_b",
        "name": "Cattle A (Profile 2 - Same Animal)",
        "folders": ["muzzleVideo2_frangi_adaptive", "1793364192_muzzleVideo2_frangi_adaptive", "muzzleVideo2"],
        "video_path": "uploads/1793364192_muzzleVideo2.mp4",
        "notes": "Alternative Take",
        "tag": "US-94210-B",
        "breed": "Holstein Friesian",
        "registered_date": "2024-09-12",
        "meta": {"source_video": "muzzleVideo2.mp4", "breed": "Holstein Friesian", "registration_id": "RF-9021", "animal_id": "cattle_001"},
    },
    {
        "id": "cattle_002",
        "name": "Cattle B (Different Animal)",
        "folders": ["muzzleVideo3_frangi_adaptive", "1793364199_muzzleVideo3_frangi_adaptive", "muzzleVideo3"],
        "video_path": "uploads/1793364199_muzzleVideo3.mp4",
        "notes": "Unrelated Dairy Cow",
        "tag": "US-88412-C",
        "breed": "Jersey",
        "registered_date": "2024-09-15",
        "meta": {"source_video": "muzzleVideo3.mp4", "breed": "Jersey", "registration_id": "RF-4412", "animal_id": "cattle_002"},
    },
    {
        "id": "cattle_003",
        "name": "Cattle C (Heifer)",
        "folders": ["1793369778_muzzleVideo4_frangi_adaptive", "muzzleVideo4_frangi_adaptive"],
        "video_path": "uploads/1793369778_muzzleVideo4.mp4",
        "notes": "Registered Heifer",
        "tag": "US-77201-D",
        "breed": "Angus",
        "registered_date": "2024-09-20",
        "meta": {"source_video": "muzzleVideo4.mp4", "breed": "Angus", "registration_id": "RF-7720", "animal_id": "cattle_003"},
    },
    {
        "id": "cattle_004",
        "name": "Cattle D (Calf)",
        "folders": ["1793369565_muzzleVideo5_frangi_adaptive", "1793369773_muzzleVideo5_frangi_adaptive", "muzzleVideo5_frangi_adaptive"],
        "video_path": "uploads/1793369565_muzzleVideo5.mp4",
        "notes": "Pasture Herd #4",
        "tag": "US-31085-E",
        "breed": "Guernsey",
        "registered_date": "2024-10-01",
        "meta": {"source_video": "muzzleVideo5.mp4", "breed": "Guernsey", "registration_id": "RF-3108", "animal_id": "cattle_004"},
    },
    {
        "id": "cattle_005",
        "name": "Cattle E (Crossbreed)",
        "folders": ["1793364507_muzzleVideo7_frangi_adaptive", "muzzleVideo7_frangi_adaptive"],
        "video_path": "uploads/1793364507_muzzleVideo7.mp4",
        "notes": "Breeding Stock",
        "tag": "US-88540-F",
        "breed": "Simmental Cross",
        "registered_date": "2024-10-05",
        "meta": {"source_video": "muzzleVideo7.mp4", "breed": "Simmental Cross", "registration_id": "RF-8854", "animal_id": "cattle_005"},
    },
]

_CACHED_GALLERY_PRESETS: Optional[List[Dict[str, Any]]] = None


def load_gallery_presets_data(force_reload: bool = False) -> List[Dict[str, Any]]:
    global _CACHED_GALLERY_PRESETS
    if _CACHED_GALLERY_PRESETS is not None and not force_reload:
        return _CACHED_GALLERY_PRESETS

    presets_json_path = BASE_DIR / "config" / "gallery_presets.json"
    if presets_json_path.exists():
        try:
            preset_content = json.loads(presets_json_path.read_text(encoding="utf-8"))
            gallery = preset_content.get("gallery", [])
            if gallery and len(gallery) > 0:
                _CACHED_GALLERY_PRESETS = gallery
                return gallery
        except Exception as exc:
            logger.warning(f"Failed loading presets JSON from {presets_json_path}: {exc}")

    vff_dir = OUTPUT_DIR / "video_five_frames"
    gallery = []

    for pdef in GALLERY_PRESET_DEFINITIONS:
        cid = pdef["id"]
        name = pdef["name"]
        notes = pdef.get("notes")
        meta = dict(pdef.get("meta") or {})
        emb = None

        # 1. Search preset folders
        for folder_name in pdef.get("folders", []):
            summary_path = vff_dir / folder_name / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    emb = _extract_embedding_from_summary(data)
                    if emb:
                        meta["source_summary"] = str(summary_path.relative_to(BASE_DIR))
                        break
                except Exception as exc:
                    logger.debug(f"Could not load {summary_path}: {exc}")

        # 2. Search any summary in vff_dir matching the stem
        if emb is None:
            stem = pdef.get("video_path", "").split("/")[-1].replace(".mp4", "")
            if stem:
                for cand_summary in sorted(vff_dir.glob(f"*{stem}*/summary.json")):
                    try:
                        data = json.loads(cand_summary.read_text(encoding="utf-8"))
                        emb = _extract_embedding_from_summary(data)
                        if emb:
                            meta["source_summary"] = str(cand_summary.relative_to(BASE_DIR))
                            break
                    except Exception:
                        pass

        if emb:
            gallery.append({
                "id": cid,
                "name": name,
                "embedding": emb,
                "notes": notes,
                "tag": pdef.get("tag"),
                "breed": pdef.get("breed"),
                "registered_date": pdef.get("registered_date"),
                "meta": meta,
            })

    _CACHED_GALLERY_PRESETS = gallery
    return gallery
