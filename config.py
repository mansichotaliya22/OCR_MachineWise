"""
config.py
=========
Centralised configuration for the Industrial OCR Monitoring System.

All runtime settings are read from environment variables (or a .env file).
Defaults are safe for local development; override them in docker-compose.yml
or a real .env for production.

Design principles
-----------------
- Single source of truth: every module imports from here — no scattered
  hardcoded strings anywhere else in the codebase.
- Pydantic BaseSettings validates types at startup, so a bad env variable
  crashes immediately with a clear message rather than failing silently.
- Paths are resolved to absolute strings so they work regardless of the
  working directory the process is launched from.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ---------------------------------------------------------------------------
# Resolve the project root once (same directory as this file)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    All application settings with types, defaults, and descriptions.
    Environment variables are case-insensitive (Pydantic handles that).
    """

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    APP_NAME: str = Field(default="Industrial OCR Monitoring System")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)

    # -----------------------------------------------------------------------
    # Paths  (resolved relative to project root)
    # -----------------------------------------------------------------------
    DATASET_PATH: str = Field(default=str(ROOT_DIR / "master_dataset" / "data.yaml"))
    MODEL_SAVE_PATH: str = Field(default=str(ROOT_DIR / "detector" / "best.pt"))
    IMAGES_INPUT_DIR: str = Field(default=str(ROOT_DIR / "images" / "input"))
    RESULTS_DIR: str = Field(default=str(ROOT_DIR / "results"))
    MODELS_DIR: str = Field(default=str(ROOT_DIR / "models"))

    # -----------------------------------------------------------------------
    # YOLOv8 Training hyper-parameters
    # -----------------------------------------------------------------------
    YOLO_BASE_MODEL: str = Field(default="yolov8n.pt")   # nano — fast to train
    YOLO_EPOCHS: int = Field(default=100)
    YOLO_IMG_SIZE: int = Field(default=640)
    YOLO_BATCH: int = Field(default=16)
    YOLO_WORKERS: int = Field(default=4)
    YOLO_PROJECT: str = Field(default=str(ROOT_DIR / "models"))
    YOLO_RUN_NAME: str = Field(default="ocr_detector")

    # YOLOv8 inference
    YOLO_CONF_THRESHOLD: float = Field(default=0.25)   # minimum detection confidence
    YOLO_IOU_THRESHOLD: float = Field(default=0.45)

    # -----------------------------------------------------------------------
    # PaddleOCR
    # -----------------------------------------------------------------------
    OCR_LANG: str = Field(default="en")
    OCR_USE_GPU: bool = Field(default=False)
    OCR_USE_ANGLE_CLS: bool = Field(default=True)   # auto-correct rotated text

    # -----------------------------------------------------------------------
    # MQTT / EMQX
    # -----------------------------------------------------------------------
    MQTT_BROKER_HOST: str = Field(default="emqx")      # service name in Docker
    MQTT_BROKER_PORT: int = Field(default=1883)
    MQTT_USERNAME: str = Field(default="")
    MQTT_PASSWORD: str = Field(default="")
    MQTT_CLIENT_ID_PUB: str = Field(default="ocr_publisher")
    MQTT_CLIENT_ID_SUB: str = Field(default="ocr_subscriber")
    MQTT_QOS: int = Field(default=1)
    MQTT_KEEPALIVE: int = Field(default=60)

    # Topics
    MQTT_TOPIC_IMAGE: str = Field(default="industrial/ocr/image")
    MQTT_TOPIC_RESULTS: str = Field(default="industrial/ocr/results")
    MQTT_TOPIC_STATUS: str = Field(default="industrial/ocr/status")

    # -----------------------------------------------------------------------
    # MongoDB
    # -----------------------------------------------------------------------
    MONGO_URI: str = Field(default="mongodb://mongo:27017")   # Docker service
    MONGO_DB_NAME: str = Field(default="ocr_db")
    MONGO_COLLECTION: str = Field(default="ocr_results")

    # -----------------------------------------------------------------------
    # FastAPI / Uvicorn
    # -----------------------------------------------------------------------
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=False)   # True only for dev

    # CORS — list of allowed origins (comma-separated in env)
    CORS_ORIGINS: str = Field(default="*")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default=str(ROOT_DIR / "results" / "app.log"))

    class Config:
        env_file = ".env"          # automatically load .env if present
        env_file_encoding = "utf-8"
        case_sensitive = False     # MY_VAR == my_var in env


# ---------------------------------------------------------------------------
# Singleton — import `settings` everywhere instead of instantiating again
# ---------------------------------------------------------------------------
settings = Settings()


# ---------------------------------------------------------------------------
# Helper: ensure all runtime directories exist
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """
    Create any directories that the application writes to at runtime.
    Call this once at application startup (e.g. FastAPI lifespan event).
    """
    dirs = [
        settings.IMAGES_INPUT_DIR,
        settings.RESULTS_DIR,
        settings.MODELS_DIR,
        str(ROOT_DIR / "detector"),
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
