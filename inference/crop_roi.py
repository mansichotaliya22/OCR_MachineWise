"""
inference/crop_roi.py
=====================
Crop detected bounding-box regions from an image.

Responsibilities
----------------
- Accept the original image (NumPy array) and a list of DetectionResult
  objects from YOLOv8.
- Return a list of cropped sub-images, one per bounding box.
- Apply a configurable padding so the crop includes a few pixels of
  context around the engraved code (improves OCR accuracy).
- Validate coordinates against image boundaries to prevent out-of-bounds
  slicing.

Public API
----------
    from inference.crop_roi import crop_regions

    crops = crop_regions(image, detections, padding=10)
    # crops[i] corresponds to detections[i]
"""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector.predict import DetectionResult


def crop_regions(
    image: np.ndarray,
    detections: List[DetectionResult],
    padding: int = 10,
) -> List[np.ndarray]:
    """
    Crop each detected region from the image.

    Parameters
    ----------
    image : np.ndarray
        Full original image (BGR or grayscale).
    detections : List[DetectionResult]
        Bounding-box detections from Detector.run().
    padding : int
        Extra pixels to include around each bounding box.
        Clamped to image boundaries automatically.

    Returns
    -------
    List[np.ndarray]
        Cropped sub-images in the same order as *detections*.
        Empty list if *detections* is empty.
    """
    if not detections:
        logger.debug("No detections to crop.")
        return []

    h_img, w_img = image.shape[:2]
    crops: List[np.ndarray] = []

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox

        # Apply padding and clamp to image bounds
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w_img, x2 + padding)
        y2p = min(h_img, y2 + padding)

        crop = image[y1p:y2p, x1p:x2p]

        if crop.size == 0:
            logger.warning(
                f"Crop [{idx}] is empty after clamping — bbox={det.bbox}, "
                f"image=({w_img}x{h_img}). Skipping."
            )
            continue

        logger.debug(
            f"Crop [{idx}]: bbox={det.bbox} → padded=({x1p},{y1p},{x2p},{y2p}) "
            f"shape={crop.shape}"
        )
        crops.append(crop)

    logger.info(f"Cropped {len(crops)} region(s) from image.")
    return crops


def save_crops(
    crops: List[np.ndarray],
    output_dir: str | Path,
    prefix: str = "crop",
) -> List[Path]:
    """
    Save cropped images to disk (useful for debugging).

    Returns list of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for i, crop in enumerate(crops):
        path = out_dir / f"{prefix}_{i:03d}.jpg"
        cv2.imwrite(str(path), crop)
        logger.debug(f"Saved crop → {path}")
        paths.append(path)

    return paths
