"""
File Upload Route
=================
Handles media file uploads (images and videos) for cattle muzzle processing.
"""

import time
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..helpers import UPLOADS_DIR

router = APIRouter()


@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a video or image file for muzzle cropping."""
    filename = Path(file.filename).name
    save_path = UPLOADS_DIR / f"{int(time.time())}_{filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = save_path.stat().st_size
    if file_size == 0:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{filename}' is empty (0 bytes). Please ensure the file has completed downloading and is not corrupted."
        )

    is_video = save_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    return {
        "success": True,
        "filename": filename,
        "path": str(save_path),
        "is_video": is_video,
        "size_mb": round(file_size / (1024 * 1024), 2),
    }
