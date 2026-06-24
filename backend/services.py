"""
backend/services.py
===================
Business logic layer between FastAPI routes and the underlying
OCR pipeline, MQTT publisher, and MongoDB.

Keeping logic here (not in routes.py) means:
- Routes stay thin: validate input → call service → return response.
- Services are easily unit-testable without HTTP.
- Dependencies (pipeline, publisher, DB) are injected and swappable.

Functions
---------
- process_image(file_bytes, filename) → OCRResultResponse
    Run the full pipeline and persist + publish the result.
- get_latest()                        → OCRResultResponse | None
- get_history(limit)                  → HistoryResponse
- delete_history()                    → DeleteResponse
- search_by_filename(name)            → HistoryResponse
"""

import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from database.mongo import (
    delete_all_history,
    find_by_filename,
    get_history,
    get_latest_result,
    insert_result,
)
from inference.pipeline import OCRPipeline
from mqtt.publisher import MQTTPublisher

from backend.schemas import (
    DeleteResponse,
    HistoryResponse,
    OCRResultResponse,
)


# ---------------------------------------------------------------------------
# Singleton objects — initialised once at application startup
# ---------------------------------------------------------------------------
_pipeline: Optional[OCRPipeline] = None
_publisher: Optional[MQTTPublisher] = None


def get_pipeline() -> OCRPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising OCR pipeline (first request)...")
        _pipeline = OCRPipeline()
    return _pipeline


def get_publisher() -> MQTTPublisher:
    global _publisher
    if _publisher is None:
        logger.info("Initialising MQTT publisher...")
        _publisher = MQTTPublisher()
        try:
            _publisher.connect(timeout=5)
        except Exception as e:
            logger.warning(f"MQTT publisher could not connect: {e}. Results will not be published.")
    return _publisher


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

async def process_image(file_bytes: bytes, filename: str) -> OCRResultResponse:
    """
    Run the full OCR pipeline on uploaded image bytes.

    Steps:
    1. Decode bytes → NumPy BGR array.
    2. Save the raw upload to images/input/.
    3. Run OCRPipeline.
    4. Persist result to MongoDB.
    5. Publish result to MQTT.
    6. Return OCRResultResponse.
    """
    # ---- Decode image --------------------------------------------------
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {filename}")

    # ---- Save original upload ------------------------------------------
    input_dir = Path(settings.IMAGES_INPUT_DIR)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / filename
    cv2.imwrite(str(input_path), image)
    logger.info(f"Saved upload → {input_path}")

    # ---- Save annotated image path -------------------------------------
    results_dir = Path(settings.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run pipeline --------------------------------------------------
    pipeline = get_pipeline()
    ocr_result, annotated = pipeline.run(
        source=image,
        save_annotated=True,
        output_dir=results_dir,
    )
    # Overwrite filename stem to match the upload filename
    stem = Path(filename).stem
    ocr_result.filename = stem

    # Save annotated image with correct name
    ann_path = results_dir / f"{stem}_det.jpg"
    cv2.imwrite(str(ann_path), annotated)
    logger.info(f"Annotated image saved → {ann_path}")

    # ---- Persist to MongoDB -------------------------------------------
    timestamp = datetime.now(timezone.utc)
    doc = {
        "filename":   ocr_result.filename,
        "text":       ocr_result.text,
        "length":     ocr_result.length,
        "confidence": ocr_result.confidence,
        "status":     ocr_result.status,
        "timestamp":  timestamp,
    }
    doc_id = await insert_result(doc)

    # ---- Publish via MQTT ---------------------------------------------
    try:
        publisher = get_publisher()
        publisher.publish_result(ocr_result)
    except Exception as e:
        logger.warning(f"MQTT publish skipped: {e}")

    return OCRResultResponse(
        id=doc_id,
        filename=ocr_result.filename,
        text=ocr_result.text,
        length=ocr_result.length,
        confidence=ocr_result.confidence,
        status=ocr_result.status,
        timestamp=timestamp,
    )


async def get_latest() -> Optional[OCRResultResponse]:
    """Return the most recent OCR result from MongoDB."""
    doc = await get_latest_result()
    if not doc:
        return None
    return OCRResultResponse(
        id=doc.get("_id"),
        filename=doc.get("filename", ""),
        text=doc.get("text", ""),
        length=doc.get("length", 0),
        confidence=doc.get("confidence", 0.0),
        status=doc.get("status", ""),
        timestamp=doc.get("timestamp"),
    )


async def fetch_history(limit: int = 50) -> HistoryResponse:
    """Return the last *limit* OCR results."""
    docs = await get_history(limit)
    results = [
        OCRResultResponse(
            id=d.get("_id"),
            filename=d.get("filename", ""),
            text=d.get("text", ""),
            length=d.get("length", 0),
            confidence=d.get("confidence", 0.0),
            status=d.get("status", ""),
            timestamp=d.get("timestamp"),
        )
        for d in docs
    ]
    return HistoryResponse(total=len(results), results=results)


async def remove_history() -> DeleteResponse:
    """Delete all OCR results from MongoDB."""
    count = await delete_all_history()
    return DeleteResponse(deleted=count, message=f"Deleted {count} record(s).")


async def search_filename(filename: str) -> HistoryResponse:
    """Search OCR results by filename."""
    docs = await find_by_filename(filename)
    results = [
        OCRResultResponse(
            id=d.get("_id"),
            filename=d.get("filename", ""),
            text=d.get("text", ""),
            length=d.get("length", 0),
            confidence=d.get("confidence", 0.0),
            status=d.get("status", ""),
            timestamp=d.get("timestamp"),
        )
        for d in docs
    ]
    return HistoryResponse(total=len(results), results=results)
