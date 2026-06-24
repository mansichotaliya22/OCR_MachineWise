"""
inference/pipeline.py
=====================
End-to-end OCR pipeline: image → YOLOv8 → preprocess → PaddleOCR → result.

Pipeline stages
---------------
1.  Load image (file path or NumPy array).
2.  Run YOLOv8 to detect engraved-code regions.
3.  Crop each detected ROI (with padding).
4.  Apply OpenCV preprocessing to each ROI.
5.  Run PaddleOCR on the preprocessed ROI.
6.  Aggregate text, compute length and confidence.
7.  Return a structured OCRResult and the annotated image.

Design decisions
----------------
- PaddleOCR is initialised once (expensive) and reused across calls.
- When multiple ROIs are detected, the text with the highest aggregate
  confidence is returned as the primary result; all detections are
  included in the ``all_regions`` list.
- Confidence is averaged over all OCR text lines in a region.
- The annotated image (with YOLOv8 boxes drawn) is returned so the
  caller can save or stream it to the dashboard.

Public API
----------
    from inference.pipeline import OCRPipeline, OCRResult

    pipeline = OCRPipeline()
    result, annotated = pipeline.run("images/input/part.jpg")
    print(result.text, result.confidence)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from detector.predict import Detector, DetectionResult
from inference.crop_roi import crop_regions
from preprocessing.preprocess import Preprocessor


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------
@dataclass
class RegionOCRResult:
    """OCR result for a single detected bounding-box region."""
    region_index: int
    bbox: Tuple[int, int, int, int]
    text: str
    length: int
    confidence: float          # mean OCR confidence for this region
    detection_confidence: float  # YOLOv8 detection confidence


@dataclass
class OCRResult:
    """
    Aggregated OCR result for one image.

    This is the object published to MQTT and stored in MongoDB.
    """
    filename: str
    text: str                  # best / primary OCR text
    length: int
    confidence: float          # OCR confidence of the primary text
    all_regions: List[RegionOCRResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    status: str = "success"    # "success" | "no_detection" | "no_text" | "error"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class OCRPipeline:
    """
    Full OCR inference pipeline.

    Parameters
    ----------
    detector : Detector | None
        Pre-instantiated YOLOv8 Detector.  Created automatically if None.
    preprocessor : Preprocessor | None
        Pre-instantiated Preprocessor.  Created automatically if None.
    ocr_lang : str
        Language for PaddleOCR (default from config).
    use_gpu : bool
        Use GPU for PaddleOCR (default from config).
    """

    def __init__(
        self,
        detector: Optional[Detector] = None,
        preprocessor: Optional[Preprocessor] = None,
        ocr_lang: str = settings.OCR_LANG,
        use_gpu: bool = settings.OCR_USE_GPU,
    ) -> None:
        logger.info("Initialising OCR Pipeline...")

        self._detector = detector or Detector()
        self._preprocessor = preprocessor or Preprocessor()
        self._ocr = self._init_paddleocr(ocr_lang, use_gpu)

        logger.success("OCR Pipeline ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _init_paddleocr(lang: str, use_gpu: bool) -> PaddleOCR:
        """
        Initialise PaddleOCR once.

        PaddleOCR downloads model weights on first run (~50 MB).
        Subsequent runs use the cached models.
        """
        logger.info(f"Loading PaddleOCR (lang={lang}, gpu={use_gpu})...")
        try:
            ocr = PaddleOCR(
                use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,       # suppress PaddlePaddle internal logs
            )
            logger.success("PaddleOCR loaded.")
            return ocr
        except Exception as exc:
            logger.error(f"PaddleOCR init failed: {exc}")
            raise RuntimeError(f"PaddleOCR init failed: {exc}") from exc

    @staticmethod
    def _load_image(source: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, str]:
        """
        Return (bgr_image, filename_stem).
        """
        if isinstance(source, np.ndarray):
            return source.copy(), "frame"

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img, path.stem

    def _run_ocr_on_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Run PaddleOCR on a single (possibly preprocessed) crop.

        Returns
        -------
        text : str
            All recognised text lines joined by a space.
        confidence : float
            Mean confidence across all recognised lines.
        """
        # PaddleOCR accepts BGR or grayscale NumPy array
        try:
            ocr_result = self._ocr.ocr(crop, cls=settings.OCR_USE_ANGLE_CLS)
        except Exception as exc:
            logger.warning(f"PaddleOCR raised an exception: {exc}")
            return "", 0.0

        if not ocr_result or ocr_result[0] is None:
            return "", 0.0

        texts: List[str] = []
        confidences: List[float] = []

        for line in ocr_result[0]:
            if line is None:
                continue
            # line format: [[bbox_points], (text, confidence)]
            try:
                text_part = line[1][0].strip()
                conf_part = float(line[1][1])
                if text_part:
                    texts.append(text_part)
                    confidences.append(conf_part)
            except (IndexError, TypeError, ValueError) as e:
                logger.debug(f"Malformed OCR line, skipping: {e}")
                continue

        if not texts:
            return "", 0.0

        combined_text = " ".join(texts)
        mean_conf = round(sum(confidences) / len(confidences), 4)
        return combined_text, mean_conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        source: Union[str, Path, np.ndarray],
        roi_padding: int = 10,
        save_annotated: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[OCRResult, np.ndarray]:
        """
        Execute the full pipeline on one image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Input image.
        roi_padding : int
            Pixel padding around each detected bounding box before OCR.
        save_annotated : bool
            Save the annotated (boxes-drawn) image to disk.
        output_dir : str | Path | None
            Directory for annotated images.  Defaults to settings.RESULTS_DIR.

        Returns
        -------
        OCRResult
            Structured result ready for MQTT / MongoDB.
        annotated_image : np.ndarray
            BGR image with detection boxes drawn.
        """
        t_start = time.perf_counter()

        # ---- Load ---------------------------------------------------------
        image, filename = self._load_image(source)
        logger.info(f"Pipeline started for: {filename}")

        # ---- Detect -------------------------------------------------------
        detections, annotated = self._detector.run(
            image,
            save_annotated=save_annotated,
            output_path=(
                Path(output_dir or settings.RESULTS_DIR) / f"{filename}_det.jpg"
                if save_annotated else None
            ),
        )

        if not detections:
            logger.warning(f"No engraved-code regions detected in: {filename}")
            elapsed = (time.perf_counter() - t_start) * 1000
            return (
                OCRResult(
                    filename=filename,
                    text="",
                    length=0,
                    confidence=0.0,
                    processing_time_ms=round(elapsed, 2),
                    status="no_detection",
                ),
                annotated,
            )

        # ---- Crop ---------------------------------------------------------
        crops = crop_regions(image, detections, padding=roi_padding)

        # ---- Preprocess + OCR per region ----------------------------------
        region_results: List[RegionOCRResult] = []

        for idx, (crop, det) in enumerate(zip(crops, detections)):
            # Preprocess
            try:
                preproc_stages = self._preprocessor.run(crop)
                final_crop = preproc_stages["final"]
            except Exception as e:
                logger.warning(f"Preprocessing failed for region {idx}: {e} — using raw crop")
                final_crop = crop

            # OCR
            text, conf = self._run_ocr_on_crop(final_crop)
            logger.info(f"Region [{idx}] text='{text}'  conf={conf:.4f}")

            region_results.append(
                RegionOCRResult(
                    region_index=idx,
                    bbox=det.bbox,
                    text=text,
                    length=len(text),
                    confidence=conf,
                    detection_confidence=det.confidence,
                )
            )

        # ---- Aggregate ----------------------------------------------------
        # Pick the region with the highest OCR confidence as the primary result
        valid_regions = [r for r in region_results if r.text]
        if not valid_regions:
            elapsed = (time.perf_counter() - t_start) * 1000
            return (
                OCRResult(
                    filename=filename,
                    text="",
                    length=0,
                    confidence=0.0,
                    all_regions=region_results,
                    processing_time_ms=round(elapsed, 2),
                    status="no_text",
                ),
                annotated,
            )

        best = max(valid_regions, key=lambda r: r.confidence)

        elapsed = (time.perf_counter() - t_start) * 1000
        result = OCRResult(
            filename=filename,
            text=best.text,
            length=best.length,
            confidence=best.confidence,
            all_regions=region_results,
            processing_time_ms=round(elapsed, 2),
            status="success",
        )

        logger.success(
            f"Pipeline done  text='{result.text}'  "
            f"len={result.length}  conf={result.confidence}  "
            f"time={elapsed:.1f}ms"
        )
        return result, annotated


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run OCR pipeline on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", action="store_true", help="Save annotated image")
    args = parser.parse_args()

    pipeline = OCRPipeline()
    ocr_result, ann_img = pipeline.run(args.image, save_annotated=args.save)

    print(json.dumps({
        "filename": ocr_result.filename,
        "text": ocr_result.text,
        "length": ocr_result.length,
        "confidence": ocr_result.confidence,
        "status": ocr_result.status,
        "processing_time_ms": ocr_result.processing_time_ms,
    }, indent=2))
