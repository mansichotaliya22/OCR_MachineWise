"""
detector/train.py
=================
YOLOv8 training script for engraved-code detection.

Usage
-----
Run from the project root:

    python -m detector.train

Or directly:

    python detector/train.py

What it does
------------
1. Reads all hyper-parameters from config.py (single source of truth).
2. Verifies the dataset YAML exists before starting.
3. Trains YOLOv8n on master_dataset/ for 100 epochs.
4. Copies the best checkpoint to detector/best.pt so every other module
   can find it at a well-known path.
5. Logs progress through loguru (goes to stdout + results/app.log).
6. Handles keyboard interrupts gracefully — partial training is saved.

Architecture note
-----------------
Training is intentionally kept in its own module so it can be triggered
via CLI, a Jupyter notebook, or a management endpoint without touching
the inference or serving code.
"""

import shutil
import sys
from pathlib import Path

from loguru import logger
from ultralytics import YOLO

# ---- project imports -------------------------------------------------------
# Allow running as `python detector/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings, ensure_directories, ROOT_DIR


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    """Attach a file sink in addition to the default stderr sink."""
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> Path:
    """
    Train YOLOv8 and return the path to the saved best.pt.

    Returns
    -------
    Path
        Absolute path to detector/best.pt after training.

    Raises
    ------
    FileNotFoundError
        If the dataset YAML specified in config.py does not exist.
    RuntimeError
        If training completes but the best.pt checkpoint is missing
        (indicates a training failure).
    """
    _configure_logging()
    ensure_directories()

    # ---- Validate dataset --------------------------------------------------
    dataset_yaml = Path(settings.DATASET_PATH)
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at: {dataset_yaml}\n"
            "Place your master_dataset/ folder in the project root and make "
            "sure data.yaml is present."
        )

    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME}  v{settings.APP_VERSION}")
    logger.info("  YOLOv8 Training")
    logger.info("=" * 60)
    logger.info(f"Dataset  : {dataset_yaml}")
    logger.info(f"Base model: {settings.YOLO_BASE_MODEL}")
    logger.info(f"Epochs   : {settings.YOLO_EPOCHS}")
    logger.info(f"Img size : {settings.YOLO_IMG_SIZE}")
    logger.info(f"Batch    : {settings.YOLO_BATCH}")
    logger.info(f"Workers  : {settings.YOLO_WORKERS}")
    logger.info(f"Project  : {settings.YOLO_PROJECT}")
    logger.info(f"Run name : {settings.YOLO_RUN_NAME}")
    logger.info("-" * 60)

    # ---- Load base model ---------------------------------------------------
    logger.info(f"Loading base model: {settings.YOLO_BASE_MODEL}")
    model = YOLO(settings.YOLO_BASE_MODEL)

    # ---- Train -------------------------------------------------------------
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=settings.YOLO_EPOCHS,
            imgsz=settings.YOLO_IMG_SIZE,
            batch=settings.YOLO_BATCH,
            workers=settings.YOLO_WORKERS,
            project=settings.YOLO_PROJECT,
            name=settings.YOLO_RUN_NAME,
            exist_ok=True,         # overwrite previous run of same name
            verbose=True,
            # Augmentation — keep defaults; good for industrial imagery
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,          # slight rotation for engraved codes
            translate=0.1,
            scale=0.5,
            flipud=0.0,            # upside-down flip rarely valid for codes
            fliplr=0.5,
            mosaic=1.0,
        )
        logger.info("Training complete.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Ultralytics saves a partial checkpoint — attempting copy.")

    # ---- Locate best checkpoint --------------------------------------------
    run_dir = Path(settings.YOLO_PROJECT) / settings.YOLO_RUN_NAME
    best_ckpt = run_dir / "weights" / "best.pt"

    if not best_ckpt.exists():
        # Fall back to last.pt if best.pt doesn't exist (e.g. interrupted early)
        last_ckpt = run_dir / "weights" / "last.pt"
        if last_ckpt.exists():
            logger.warning("best.pt not found; using last.pt instead.")
            best_ckpt = last_ckpt
        else:
            raise RuntimeError(
                f"No checkpoint found in {run_dir / 'weights'}. "
                "Training may have failed before saving any weights."
            )

    # ---- Copy to canonical location ----------------------------------------
    dest = Path(settings.MODEL_SAVE_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_ckpt), str(dest))

    logger.success(f"Best model saved → {dest}")
    logger.info("=" * 60)
    return dest


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    saved_at = train()
    print(f"\n✅  Training finished.  Model saved at: {saved_at}")
