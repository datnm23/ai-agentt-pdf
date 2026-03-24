"""Image preprocessing pipeline — deskew, glare removal, binarization."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from skimage.filters import threshold_sauvola
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class PreprocessingPipeline:
    """
    Full image preprocessing pipeline for document quality enhancement.
    Handles glare, skew/tilt, blur, and binarization.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, image_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Run full pipeline on image file.
        Returns path to processed image.
        """
        if not HAS_CV2:
            logger.warning("OpenCV not available, returning original image.")
            return image_path

        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Cannot read image: {image_path}")
            return image_path

        logger.info(f"Preprocessing {image_path.name} — size {img.shape}")

        # ── Step 1: Upscale if low resolution ─────────────────
        img = self._upscale_if_needed(img)

        # ── Step 2: Remove glare / highlight ──────────────────
        img = self._remove_glare(img)

        # ── Step 3: Deskew rotation ────────────────────────────
        img = self._deskew_rotation(img)

        # ── Step 4: Perspective correction ────────────────────
        img = self._perspective_correction(img)

        # ── Step 5: Enhance contrast + denoise ────────────────
        img = self._enhance(img)

        # ── Step 6: Binarization ──────────────────────────────
        img = self._binarize(img)

        # Save result
        if output_path is None:
            output_path = image_path.parent / f"preprocessed_{image_path.name}"

        cv2.imwrite(str(output_path), img)
        logger.info(f"Preprocessed image saved: {output_path.name}")
        return output_path

    # ─────────────────────────────────────────────────────────
    # Step 1: Upscale
    # ─────────────────────────────────────────────────────────
    def _upscale_if_needed(self, img: np.ndarray, min_width: int = 1200) -> np.ndarray:
        h, w = img.shape[:2]
        if w < min_width:
            scale = min_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Upscaled to {new_w}×{new_h}")
        return img

    # ─────────────────────────────────────────────────────────
    # Step 2: Glare removal
    # ─────────────────────────────────────────────────────────
    def _remove_glare(self, img: np.ndarray) -> np.ndarray:
        try:
            # Convert to LAB and apply CLAHE on L channel
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)

            # Detect overexposed regions (glare mask)
            glare_mask = (l > 240).astype(np.uint8) * 255

            if np.sum(glare_mask) > 0:
                # Inpaint glare regions
                l_inpainted = cv2.inpaint(l, glare_mask, 3, cv2.INPAINT_NS)
                merged = cv2.merge([l_inpainted, a, b])
            else:
                merged = cv2.merge([l_eq, a, b])

            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            logger.debug("Glare removal applied.")
            return result
        except Exception as e:
            logger.warning(f"Glare removal failed: {e}")
            return img

    # ─────────────────────────────────────────────────────────
    # Step 3: Deskew rotation (Hough Lines approach)
    # ─────────────────────────────────────────────────────────
    def _deskew_rotation(self, img: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

            if lines is None:
                return img

            angles = []
            for rho, theta in lines[:, 0]:
                angle_deg = np.degrees(theta) - 90
                if abs(angle_deg) < 45:
                    angles.append(angle_deg)

            if not angles:
                return img

            median_angle = float(np.median(angles))
            if abs(median_angle) < 0.3:
                return img  # Almost straight, skip

            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            logger.debug(f"Deskewed rotation: {median_angle:.2f}°")
            return rotated

        except Exception as e:
            logger.warning(f"Deskew rotation failed: {e}")
            return img

    # ─────────────────────────────────────────────────────────
    # Step 4: Perspective correction (4-point warp)
    # ─────────────────────────────────────────────────────────
    def _perspective_correction(self, img: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blur, 30, 100)

            # Dilate to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edged, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img

            # Find largest 4-sided contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

                img_area = img.shape[0] * img.shape[1]
                cnt_area = cv2.contourArea(cnt)

                if len(approx) == 4 and cnt_area > img_area * 0.2:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    warped = self._four_point_transform(img, pts)
                    logger.debug("Perspective correction applied.")
                    return warped

            return img

        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}")
            return img

    def _four_point_transform(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 4-point perspective transform."""
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # ─────────────────────────────────────────────────────────
    # Step 5: Contrast enhance + denoise + sharpening
    # ─────────────────────────────────────────────────────────
    def _enhance(self, img: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect blur (Laplacian variance)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 100:
                # Unsharp masking for deblur
                blurred = cv2.GaussianBlur(gray, (0, 0), 3)
                gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                logger.debug(f"Sharpened (laplacian var={lap_var:.1f})")

            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return img

    # ─────────────────────────────────────────────────────────
    # Step 6: Binarization (Sauvola adaptive thresholding)
    # ─────────────────────────────────────────────────────────
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if HAS_SKIMAGE:
                # Sauvola thresholding — handles uneven lighting well
                window_size = 51
                thresh = threshold_sauvola(gray, window_size=window_size)
                binary = (gray > thresh).astype(np.uint8) * 255
            else:
                # Fallback: adaptive Gaussian threshold
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10
                )

            logger.debug("Binarization applied.")
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return img
