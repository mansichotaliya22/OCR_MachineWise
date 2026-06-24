"""
backend/main.py
===============
FastAPI application entry-point.

Responsibilities
----------------
- Create the FastAPI app instance with metadata for OpenAPI docs.
- Register CORS middleware so the browser dashboard can call the API.
- Mount the static frontend (HTML/CSS/JS) at /.
- Include the API router at /api.
- Define a lifespan context manager that:
    - Creates runtime directories.
    - Pings MongoDB to verify connectivity.
    - Starts the MQTT subscriber background thread.
    - Cleans up gracefully on shutdown.
- Run with Uvicorn when executed directly (dev mode).

Production run
--------------
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings, ensure_directories
from database.mongo import ping_database, insert_result
from mqtt.subscriber import MQTTSubscriber
from backend.routes import router


# ---------------------------------------------------------------------------
# MQTT subscriber — started in lifespan
# ---------------------------------------------------------------------------
_subscriber: MQTTSubscriber | None = None


async def _on_mqtt_result(payload: dict) -> None:
    """
    Callback invoked by the MQTT subscriber for every incoming OCR result.

    Persists the payload to MongoDB.
    (Phase 3: will also broadcast via WebSocket here.)
    """
    try:
        await insert_result(payload)
        logger.info(f"MQTT→MongoDB: stored result for '{payload.get('filename', '')}'")
    except Exception as e:
        logger.error(f"MQTT→MongoDB insert failed: {e}")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    AsyncContextManager used by FastAPI as the application lifespan.
    Code before `yield` runs on startup; code after runs on shutdown.
    """
    global _subscriber

    # ---- Startup ----------------------------------------------------------
    logger.info("=" * 55)
    logger.info(f"  {settings.APP_NAME}  v{settings.APP_VERSION}")
    logger.info("  Starting up...")
    logger.info("=" * 55)

    ensure_directories()

    # Verify MongoDB
    if await ping_database():
        logger.success("MongoDB connection OK.")
    else:
        logger.warning(
            "MongoDB unreachable at startup. Retries will happen per request."
        )

    # Start MQTT subscriber
    try:
        # Wrap async callback in a sync shim (paho callbacks are synchronous)
        import asyncio

        loop = asyncio.get_event_loop()

        def sync_callback(payload: dict):
            asyncio.run_coroutine_threadsafe(_on_mqtt_result(payload), loop)

        _subscriber = MQTTSubscriber(on_result_callback=sync_callback)
        _subscriber.start()
    except Exception as e:
        logger.warning(f"MQTT subscriber startup failed: {e}. Continuing without MQTT.")

    logger.success("Application startup complete.")

    yield   # ← application runs here

    # ---- Shutdown ---------------------------------------------------------
    logger.info("Shutting down...")
    if _subscriber:
        _subscriber.stop()
    logger.info("Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Industrial OCR Monitoring System — "
        "detects and reads engraved codes on machine parts using "
        "YOLOv8 + PaddleOCR, with MQTT and MongoDB integration."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins in dev; restrict in production via env var
origins = (
    [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    if settings.CORS_ORIGINS != "*"
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes — all under /api prefix
app.include_router(router, prefix="/api", tags=["OCR API"])

# Static frontend — served at /  (must come after API routes)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found: {frontend_dir}")


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
