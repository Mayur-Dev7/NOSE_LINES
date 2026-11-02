"""
Machine Learning & Biometric Processing Package for Bovine Muzzle Identification.
=================================================================================
Provides modular ML components for muzzle detection, segmentation, groove graph
extraction, reference template consensus, and ConvNeXt-Tiny biometric embeddings.
Submodules are available via `from ml.<submodule> import ...` or lazy attribute access.
"""

import sys
import importlib
from pathlib import Path

# Ensure project root and ml directory are in sys.path
_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_ML_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EXPORTS = {
    # Config
    "BASE_DIR": "ml.config",
    "WEIGHTS_PATH": "ml.config",
    "OUTPUT_DIR": "ml.config",
    "UPLOADS_DIR": "ml.config",
    # Schema
    "SCHEMA_VERSION": "ml.template_schema",
    "make_empty_template": "ml.template_schema",
    "robust_median_and_mad": "ml.template_schema",
    "calculate_stability": "ml.template_schema",
    # Groove Extraction
    "extract_muzzle_grooves_pipeline": "ml.groove_extractor",
    # Groove Graph
    "extract_skeleton_graph": "ml.groove_graph",
    "visualize_groove_graph": "ml.groove_graph",
    "prune_skeleton": "ml.groove_graph",
    "extract_minutiae_points": "ml.groove_graph",
    # Batch Processing
    "process_image_batch": "ml.batch_processor",
    # Consensus
    "build_consensus_template": "ml.template_consensus",
    # Template Matcher
    "save_reference_template": "ml.template_matcher",
    "load_reference_template": "ml.template_matcher",
    "list_saved_templates": "ml.template_matcher",
    "compare_two_templates": "ml.template_matcher",
    # Template Exporter
    "export_all_template_reports": "ml.template_exporter",
    "export_consensus_summary_csv": "ml.template_exporter",
    "export_per_image_measurements_csv": "ml.template_exporter",
    "export_template_json": "ml.template_exporter",
    # Video Groove Extractor
    "extract_five_frames_groove_features": "ml.video_groove_extractor",
    "compute_cosine_similarity": "ml.video_groove_extractor",
    # Muzzle Matcher
    "enhance_muzzle_image": "ml.muzzle_matcher",
    "compare_two_muzzles": "ml.muzzle_matcher",
    "match_sift_ransac": "ml.muzzle_matcher",
    "match_minutiae": "ml.muzzle_matcher",
    "match_muzzle_frame_sets": "ml.muzzle_matcher",
    "match_two_videos_end_to_end": "ml.muzzle_matcher",
    "load_matcher_config": "ml.muzzle_matcher",
    "match_video_sets": "ml.muzzle_matcher",
    "match_spatial_video_sets": "ml.muzzle_matcher",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
