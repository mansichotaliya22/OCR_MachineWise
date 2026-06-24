"""
preprocessing/preprocess.py
============================
OpenCV-based image preprocessing pipeline for engraved-code OCR.

Why preprocessing matters here
-------------------------------
Engraved / laser-etched codes have very low contrast against the metal
surface and are highly sensitive to lighting conditions.  A robust
preprocessing chain (CLAHE → denoise → sharpen → adaptive threshold)
dramatically increases PaddleOCR accuracy on these images.

Each step is a pure function that accepts and returns a NumPy array so
they can be chained, tested independently, or skipped.

Public API
----------
    from preprocessing.preprocess import Preprocessor

    prep = Preprocessor()
    result = prep.run(image_bgr)   # returns dict with intermediate images
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ---------------------------------------------------------------------------
# Individual processing functions  (pure, stateless)
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or already-gray image to single-channel grayscale."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.

    Enhances local contrast without over-amplifying noise — ideal for
    low-contrast engravings under variable industrial lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def histogram_equalise(gray: np.ndarray) -> np.ndarray:
    """
    Global histogram equalisation.

    Applied *after* CLAHE for an additional global contrast boost.
    Use with caution on very noisy images; CLAHE alone is usually enough.
    """
    return cv2.equalizeHist(gray)


def gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    sigma: float = 0,
) -> np.ndarray:
    """Light Gaussian blur to suppress high-frequency sensor noise."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def denoise(
    image: np.ndarray,
    h: float = 10,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising (fastNlMeansDenoising).

    More aggressive than Gaussian blur but preserves edges better —
    important for character strokes in engraved text.

    Parameters
    ----------
    h : float
        Filter strength.  Higher → smoother but loses fine detail.
    """
    return cv2.fastNlMeansDenoising(
        image, None, h, template_window, search_window
    )


def adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """
    Adaptive (local) binarisation using Gaussian-weighted neighbourhood.

    Handles non-uniform illumination far better than a global threshold,
    which is essential for curved or reflective metal surfaces.

    block_size must be odd and > 1.
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Unsharp-mask sharpening via a Laplacian kernel.

    Enhances edge definition of character strokes before OCR.
    """
    kernel = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast(
    image: np.ndarray,
    alpha: float = 1.5,
    beta: float = 0,
) -> np.ndarray:
    """
    Linear contrast stretch: output = alpha * pixel + beta.

    alpha > 1 increases contrast; beta shifts brightness.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Preprocessor class — orchestrates the full pipeline
# ---------------------------------------------------------------------------
@dataclass
class PreprocessorConfig:
    """Tuneable knobs for the preprocessing pipeline."""
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int, int] = (8, 8)
    blur_kernel: Tuple[int, int] = (3, 3)
    denoise_h: float = 10
    adaptive_block: int = 11
    adaptive_c: int = 2
    contrast_alpha: float = 1.5
    contrast_beta: float = 0
    use_histogram_eq: bool = False   # off by default — CLAHE is sufficient
    use_adaptive_thresh: bool = True


class Preprocessor:
    """
    Full preprocessing pipeline for engraved-code ROI images.

    Returns a dict of intermediate images so callers can visualise or
    debug each stage independently.

    Parameters
    ----------
    config : PreprocessorConfig | None
        Pipeline parameters.  Defaults to sensible values for industrial
        metal-surface engravings.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.cfg = config or PreprocessorConfig()
        logger.debug("Preprocessor initialised with config: {}", self.cfg)

    def run(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute the full preprocessing chain.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).

        Returns
        -------
        dict
            Keys correspond to pipeline stage names.  The final output
            for OCR is ``result["final"]``.

        Stage order
        -----------
        original → grayscale → clahe → [hist_eq] → blur →
        denoise → contrast_enhance → sharpen → [adaptive_thresh]
        """
        if image is None or image.size == 0:
            raise ValueError("Preprocessor received an empty image.")

        stages: Dict[str, np.ndarray] = {}

        try:
            stages["original"] = image.copy()
            logger.debug("Stage 1: grayscale")
            gray = to_grayscale(image)
            stages["grayscale"] = gray

            logger.debug("Stage 2: CLAHE")
            clahe_img = apply_clahe(gray, self.cfg.clahe_clip, self.cfg.clahe_tile)
            stages["clahe"] = clahe_img

            if self.cfg.use_histogram_eq:
                logger.debug("Stage 3: histogram equalisation")
                clahe_img = histogram_equalise(clahe_img)
                stages["hist_eq"] = clahe_img

            logger.debug("Stage 4: Gaussian blur")
            blurred = gaussian_blur(clahe_img, self.cfg.blur_kernel)
            stages["blur"] = blurred

            logger.debug("Stage 5: denoising")
            denoised = denoise(blurred, h=self.cfg.denoise_h)
            stages["denoised"] = denoised

            logger.debug("Stage 6: contrast enhancement")
            contrasted = enhance_contrast(
                denoised, self.cfg.contrast_alpha, self.cfg.contrast_beta
            )
            stages["contrast_enhanced"] = contrasted

            logger.debug("Stage 7: sharpening")
            sharpened = sharpen(contrasted)
            stages["sharpened"] = sharpened

            if self.cfg.use_adaptive_thresh:
                logger.debug("Stage 8: adaptive threshold")
                thresh = adaptive_threshold(
                    sharpened, self.cfg.adaptive_block, self.cfg.adaptive_c
                )
                stages["adaptive_threshold"] = thresh
                stages["final"] = thresh
            else:
                stages["final"] = sharpened

            logger.info(
                f"Preprocessing complete. Final shape: {stages['final'].shape}"
            )

        except Exception as exc:
            logger.error(f"Preprocessing failed at stage: {exc}")
            raise

        return stages


# ---------------------------------------------------------------------------
# CLI debug helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a single image (debug)")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--save-dir", default="results", help="Directory to save stages")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read {args.image}")
        sys.exit(1)

    prep = Preprocessor()
    stages_out = prep.run(img)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for stage_name, stage_img in stages_out.items():
        out_path = save_dir / f"stage_{stage_name}.jpg"
        cv2.imwrite(str(out_path), stage_img)
        print(f"Saved: {out_path}")

    print("\nDone. Check results/ for all preprocessing stages.")
