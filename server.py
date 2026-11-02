#!/usr/bin/env python3
"""
Cattle Muzzle AI Detection API — Entry Point
=============================================
Thin launcher that imports the FastAPI app from the api package
and runs it with Uvicorn.
"""
from api.app import app

# Backwards compatibility re-exports for existing tests and scripts
from api.log_handler import _PIPELINE_LOGS, PipelineLogHandler
from api.helpers import (
    resolve_video_path,
    resolve_image_path,
    _spatial_cache_key,
    parse_cpu_cores,
)
from api.routes.status import (
    get_system_status,
    get_recent_pipeline_logs,
    clear_pipeline_logs,
)
from api.routes.matching import (
    VideoMatchRequest,
    CosineSimilarityRequest,
    ExtractVideoFiveFramesRequest,
    match_video_endpoint,
    get_spatial_match_config_endpoint,
    get_matcher_config_endpoint,
    compute_cosine_similarity_endpoint,
    execute_secondary_orb_verification,
)
from api.routes.embeddings import (
    OneToNMatchRequest,
    one_to_n_match_endpoint,
    match_one_to_n_endpoint,
    get_embedding_gallery_presets_endpoint,
    get_gallery_presets_endpoint,
    SingleVideoEmbeddingExportRequest,
    export_single_video_embedding_post,
    export_single_video_embedding_get,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
