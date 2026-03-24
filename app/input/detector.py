"""Document type detector — classifies input file before processing."""
from __future__ import annotations
import io
from pathlib import Path
from typing import Tuple
import filetype
from loguru import logger

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from app.models.schemas import DocumentType


# Image MIME types accepted
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp", "image/heic"}
PDF_MIME = "application/pdf"


def detect_document_type(file_path: Path) -> Tuple[DocumentType, float]:
    """
    Classify a document into one of DocumentType values.
    Returns (DocumentType, confidence 0-1).
    """
    mime = _get_mime(file_path)
    logger.info(f"Detected MIME: {mime} for {file_path.name}")

    if mime == PDF_MIME:
        return _classify_pdf(file_path)
    elif mime in IMAGE_MIMES or file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
        return _classify_image(file_path)
    else:
        return DocumentType.UNKNOWN, 0.0


def _get_mime(file_path: Path) -> str:
    kind = filetype.guess(str(file_path))
    if kind:
        return kind.mime
    # Fallback by extension
    ext_map = {
        ".pdf": PDF_MIME,
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }
    return ext_map.get(file_path.suffix.lower(), "application/octet-stream")


def _classify_pdf(file_path: Path) -> Tuple[DocumentType, float]:
    """Detect if PDF has real text or is just scanned images."""
    if not HAS_PDFPLUMBER:
        return DocumentType.SCANNED_PDF, 0.5

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            total_chars = 0
            pages_checked = min(3, len(pdf.pages))
            for page in pdf.pages[:pages_checked]:
                text = page.extract_text() or ""
                total_chars += len(text.strip())

            avg_chars = total_chars / max(pages_checked, 1)
            if avg_chars > 50:
                return DocumentType.SOFT_PDF, 0.95
            else:
                return DocumentType.SCANNED_PDF, 0.90
    except Exception as e:
        logger.warning(f"PDF classification error: {e}")
        return DocumentType.SCANNED_PDF, 0.5


def _classify_image(file_path: Path) -> Tuple[DocumentType, float]:
    """Classify image type: scan vs phone photo vs glare."""
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(file_path))
        if img is None:
            return DocumentType.IMAGE_PHOTO, 0.5

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Glare detection: % pixels > 240 ──────────────────
        bright_ratio = np.mean(gray > 240)
        if bright_ratio > 0.15:
            return DocumentType.IMAGE_PHOTO_GLARE, 0.85

        # ── Skew detection via horizontal edge analysis ───────
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, 3.14159 / 180, 80, minLineLength=100, maxLineGap=20)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:30]:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = abs((y2 - y1) / (x2 - x1))
                    angles.append(angle)
            if angles:
                avg_skew = sum(angles) / len(angles)
                if avg_skew > 0.15:  # > ~8.5 degrees average
                    return DocumentType.IMAGE_SKEWED, 0.80

        # ── Flatness check: uniform lighting → scan ───────────
        std_dev = float(np.std(gray))
        if std_dev < 60:
            return DocumentType.IMAGE_SCAN, 0.80
        else:
            return DocumentType.IMAGE_PHOTO, 0.75

    except Exception as e:
        logger.warning(f"Image classification error: {e}")
        return DocumentType.IMAGE_PHOTO, 0.5
