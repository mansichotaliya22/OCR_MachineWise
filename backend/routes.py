"""
backend/routes.py
=================
FastAPI router — defines all HTTP endpoints.

Endpoints
---------
POST   /upload              Upload image → run OCR → return result
GET    /latest              Most recent OCR result
GET    /history             Paginated history (query param: limit)
DELETE /history             Delete all OCR history
GET    /search/{filename}   Search results by filename
GET    /health              Service health check
GET    /images/{type}/{filename}   Serve uploaded / annotated images
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from database.mongo import ping_database
from backend.schemas import (
    DeleteResponse,
    ErrorResponse,
    HistoryResponse,
    OCRResultResponse,
    StatusResponse,
)
from backend.services import (
    fetch_history,
    get_latest,
    process_image,
    remove_history,
    search_filename,
    get_pipeline,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload & process
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=OCRResultResponse,
    summary="Upload an image and run OCR",
    description="Accepts a JPG/PNG image, runs YOLOv8 + PaddleOCR, persists result to MongoDB, and publishes to MQTT.",
)
async def upload_image(file: UploadFile = File(...)):
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > 20 * 1024 * 1024:   # 20 MB limit
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    try:
        result = await process_image(contents, file.filename)
        return result
    except ValueError as e:
        logger.warning(f"Image decode error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline error during upload: {e}")
        raise HTTPException(status_code=500, detail="OCR pipeline error. Check server logs.")


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/latest",
    response_model=Optional[OCRResultResponse],
    summary="Get the most recent OCR result",
)
async def latest_result():
    result = await get_latest()
    if result is None:
        raise HTTPException(status_code=404, detail="No OCR results found yet.")
    return result


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get OCR result history",
)
async def history(limit: int = Query(default=50, ge=1, le=200)):
    return await fetch_history(limit)


@router.delete(
    "/history",
    response_model=DeleteResponse,
    summary="Delete all OCR history",
)
async def clear_history():
    return await remove_history()


@router.get(
    "/search/{filename}",
    response_model=HistoryResponse,
    summary="Search results by filename",
)
async def search(filename: str):
    if not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")
    return await search_filename(filename.strip())


# ---------------------------------------------------------------------------
# Static image serving  (for the dashboard)
# ---------------------------------------------------------------------------
@router.get(
    "/images/input/{filename}",
    summary="Serve original uploaded image",
    include_in_schema=False,
)
async def serve_input_image(filename: str):
    path = Path(settings.IMAGES_INPUT_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(path))


@router.get(
    "/images/result/{filename}",
    summary="Serve annotated detection image",
    include_in_schema=False,
)
async def serve_result_image(filename: str):
    path = Path(settings.RESULTS_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result image not found.")
    return FileResponse(str(path))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=StatusResponse,
    summary="Service health check",
)
async def health_check():
    db_ok = await ping_database()
    try:
        pipeline = get_pipeline()
        model_loaded = pipeline._detector.is_ready()
    except Exception:
        model_loaded = False

    return StatusResponse(
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        mongodb="connected" if db_ok else "disconnected",
        mqtt="connected",   # publisher status — simplified for Phase 1
        model_loaded=model_loaded,
    )
