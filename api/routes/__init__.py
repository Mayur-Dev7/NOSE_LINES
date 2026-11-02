"""
API Routes Registration
=======================
Aggregates and registers all modular route handlers onto the FastAPI application.
"""

from fastapi import FastAPI

from .status import router as status_router
from .upload import router as upload_router
from .detection import router as detection_router
from .grooves import router as grooves_router
from .templates import router as templates_router
from .matching import router as matching_router
from .embeddings import router as embeddings_router


def register_routes(app: FastAPI) -> None:
    """Registers all API route modules onto the given FastAPI application instance."""
    app.include_router(status_router)
    app.include_router(upload_router)
    app.include_router(detection_router)
    app.include_router(grooves_router)
    app.include_router(templates_router)
    app.include_router(matching_router)
    app.include_router(embeddings_router)
