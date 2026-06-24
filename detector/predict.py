"""
detector/predict.py
===================
YOLOv8 inference module — detects engraved-code regions in an image.

Responsibilities
----------------
- Load detector/best.pt once and cache it (singleton pattern).
- Accept a file path **or** a raw NumPy array as input.
- Return a list of DetectionResult dataclasses, one per bounding box,
  sorted by confidence (highest first).
- Draw annotated bounding boxes on the original image and return it.
- Never raise bare exceptions: all errors are caught, logged, and
  re-raised as typed exceptions so callers can handle them cleanly.

Usage
-----
    from detector.predict import Detector

    detector = Detector()                       # loads model once
    detections, annotated = detector.run("images/input/part.jpg")

    for d in detections:
        print(d.bbox, d.confidence, d.class_name)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

# ---- project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """
    Single bounding-box detection from YOLOv8.

    Attributes
    ----------
    bbox : Tuple[int, int, int, int]
        Bounding box in pixel coordinates: (x1, y1, x2, y2).
    confidence : float
        Detection confidence in [0, 1].
    class_id : int
        Integer class index (0 = engraved_code).
    class_name : str
        Human-readable class label.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str = "engraved_code"


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
class Detector:
    """
    Singleton-friendly YOLOv8 inference wrapper.

    Instantiate once at application startup; the loaded model is reused
    across all calls to ``run()``.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the .pt checkpoint.  Defaults to settings.MODEL_SAVE_PATH.
    conf : float
        Minimum confidence threshold for a detection to be included.
    iou : float
        IoU threshold used for NMS.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf: float = settings.YOLO_CONF_THRESHOLD,
        iou: float = settings.YOLO_IOU_THRESHOLD,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self._model: Optional[YOLO] = None
        self._model_path = Path(model_path or settings.MODEL_SAVE_PATH)
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the YOLO model from disk, with clear error messages."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 checkpoint not found at: {self._model_path}\n"
                "Run `python -m detector.train` first to train the model."
            )
        logger.info(f"Loading YOLOv8 model from: {self._model_path}")
        try:
            self._model = YOLO(str(self._model_path))
            logger.success("YOLOv8 model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load YOLOv8 model: {exc}")
            raise RuntimeError(f"Model load failed: {exc}") from exc

    @staticmethod
    def _load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Normalise the input to an OpenCV BGR NumPy array.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or already-decoded image array.

        Returns
        -------
        np.ndarray  (H, W, 3) BGR
        """
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                # Grayscale → BGR
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            return source.copy()

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(
                f"OpenCV could not read image at {path}. "
                "Check that it is a valid image file."
            )
        return img

    @staticmethod
    def _draw_annotations(
        image: np.ndarray,
        detections: List[DetectionResult],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the image.

        Returns a new array; the input is not modified.
        """
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.2f}"

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA,
            )
        return annotated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        source: Union[str, Path, np.ndarray],
        save_annotated: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[DetectionResult], np.ndarray]:
        """
        Run YOLOv8 inference on a single image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Input image.
        save_annotated : bool
            If True, write the annotated image to *output_path*.
        output_path : str | Path | None
            Where to save the annotated image.  Required when
            ``save_annotated=True``.  Defaults to results/<filename>_det.jpg.

        Returns
        -------
        detections : List[DetectionResult]
            All detections above the confidence threshold, sorted by
            confidence descending.
        annotated_image : np.ndarray
            BGR image with bounding boxes drawn.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # ---- Load image ------------------------------------------------
        image = self._load_image(source)
        logger.debug(f"Running inference on image shape: {image.shape}")

        # ---- Inference -------------------------------------------------
        try:
            yolo_results = self._model.predict(
                source=image,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"YOLOv8 inference error: {exc}")
            raise RuntimeError(f"Inference failed: {exc}") from exc

        # ---- Parse results ---------------------------------------------
        detections: List[DetectionResult] = []
        result = yolo_results[0]  # single image → single result

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, "engraved_code")

                detections.append(
                    DetectionResult(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=round(conf_val, 4),
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        # Sort by confidence — highest first
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(f"Detected {len(detections)} region(s).")
        for i, d in enumerate(detections):
            logger.debug(f"  [{i}] bbox={d.bbox}  conf={d.confidence:.4f}")

        # ---- Annotate --------------------------------------------------
        annotated = self._draw_annotations(image, detections)

        # ---- Optionally save -------------------------------------------
        if save_annotated:
            if output_path is None:
                # derive from source path
                if isinstance(source, (str, Path)):
                    stem = Path(source).stem
                else:
                    stem = "detection"
                output_path = Path(settings.RESULTS_DIR) / f"{stem}_det.jpg"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            logger.info(f"Annotated image saved → {output_path}")

        return detections, annotated

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._model is not None


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 predict (smoke-test)")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--conf", type=float, default=settings.YOLO_CONF_THRESHOLD)
    parser.add_argument("--save", action="store_true", help="Save annotated image")
    args = parser.parse_args()

    detector = Detector(conf=args.conf)
    dets, annotated_img = detector.run(args.image, save_annotated=args.save)

    print(f"\n{'─'*50}")
    print(f"  Detections found: {len(dets)}")
    for idx, d in enumerate(dets):
        print(f"  [{idx}] {d.class_name}  conf={d.confidence:.4f}  bbox={d.bbox}")
    print(f"{'─'*50}\n")
