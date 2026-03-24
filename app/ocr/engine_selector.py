"""Multi-engine OCR selector — automatically picks the best OCR engine."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

# ── Engine name → ExtractionMethod value mapping ───────────────
ENGINE_METHOD_MAP: Dict[str, str] = {
    "paddleocr":    "ocr_paddleocr",
    "easyocr":      "ocr_easyocr",
    "tesseract":    "ocr_tesseract",
    "gemini_vision": "gemini_vision",
}


def engine_to_method(engine: str) -> str:
    """Safely convert an engine name to an ExtractionMethod value.

    Handles engine names that may or may not already have the 'ocr_' prefix.
    Falls back to the raw engine name if no mapping exists.
    """
    # Strip 'ocr_' prefix if present, then look up
    key = engine.removeprefix("ocr_")
    return ENGINE_METHOD_MAP.get(key, engine)


class OCRResult:
    def __init__(self, text: str, confidence: float, engine: str):
        self.text = text
        self.confidence = confidence
        self.engine = engine

    def __repr__(self):
        return f"OCRResult(engine={self.engine}, conf={self.confidence:.2f}, chars={len(self.text)})"


class OCREngineSelector:
    """
    Smart OCR engine selector.
    Priority: PaddleOCR → EasyOCR → Tesseract → Gemini Vision
    Falls back automatically when confidence is below threshold.

    OCR engine instances are cached at class level to avoid expensive
    re-initialisation on every call.
    """

    CONFIDENCE_THRESHOLD = 0.65

    # ── Class-level engine cache (shared across all instances) ──
    _paddle_ocr: Optional["PaddleOCR"] = None   # type: ignore[name-defined]
    _easy_reader: Optional["EasyReaderWrapper"] = None  # type: ignore[name-defined]
    _cache_lock = threading.Lock()

    def __init__(self, google_api_key: Optional[str] = None):
        self._gemini_key = google_api_key
        self._available = self._detect_available_engines()
        logger.info(f"Available OCR engines: {self._available}")

    # ── Public API ─────────────────────────────────────────────
    def extract_text(self, image_path: Path) -> OCRResult:
        """Try OCR engines in priority order, return best result."""
        errors: List[str] = []

        # ── PaddleOCR ──────────────────────────────────────────
        if "paddleocr" in self._available:
            try:
                result = self._run_paddle(image_path)
                logger.info(f"PaddleOCR confidence: {result.confidence:.2f}")
                if result.confidence >= self.CONFIDENCE_THRESHOLD:
                    return result
                errors.append(f"PaddleOCR low conf: {result.confidence:.2f}")
            except Exception as e:
                errors.append(f"PaddleOCR error: {e}")

        # ── EasyOCR ─────────────────────────────────────────────
        if "easyocr" in self._available:
            try:
                result = self._run_easyocr(image_path)
                logger.info(f"EasyOCR confidence: {result.confidence:.2f}")
                if result.confidence >= self.CONFIDENCE_THRESHOLD:
                    return result
                errors.append(f"EasyOCR low conf: {result.confidence:.2f}")
            except Exception as e:
                errors.append(f"EasyOCR error: {e}")

        # ── Tesseract ───────────────────────────────────────────
        if "tesseract" in self._available:
            try:
                result = self._run_tesseract(image_path)
                logger.info(f"Tesseract confidence: {result.confidence:.2f}")
                if result.confidence >= 0.50:
                    return result
                errors.append(f"Tesseract low conf: {result.confidence:.2f}")
            except Exception as e:
                errors.append(f"Tesseract error: {e}")

        # ── Gemini Vision fallback ──────────────────────────────
        if "gemini_vision" in self._available:
            logger.info("Falling back to Gemini Vision OCR...")
            try:
                result = self._run_gemini_vision(image_path)
                logger.info(f"Gemini Vision confidence: {result.confidence:.2f}")
                return result
            except Exception as e:
                errors.append(f"Gemini Vision error: {e}")

        # Return empty if all failed
        logger.error(f"All OCR engines failed: {errors}")
        return OCRResult(text="", confidence=0.0, engine="none")

    # ── Engine detection ───────────────────────────────────────
    def _detect_available_engines(self) -> List[str]:
        available: List[str] = []
        try:
            from paddleocr import PaddleOCR as _  # noqa: F401
            available.append("paddleocr")
        except ImportError:
            pass
        try:
            import easyocr as _  # noqa: F401
            available.append("easyocr")
        except ImportError:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            available.append("tesseract")
        except Exception:
            pass
        if self._gemini_key:
            available.append("gemini_vision")
        return available

    # ── PaddleOCR (class-cached) ───────────────────────────────
    @classmethod
    def _get_paddle_ocr(cls):
        """Return a cached PaddleOCR instance (thread-safe init)."""
        if cls._paddle_ocr is not None:
            return cls._paddle_ocr
        with cls._cache_lock:
            if cls._paddle_ocr is None:   # double-check inside lock
                from paddleocr import PaddleOCR
                cls._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
                logger.info("PaddleOCR instance cached")
        return cls._paddle_ocr

    def _run_paddle(self, image_path: Path) -> OCRResult:
        ocr = self._get_paddle_ocr()
        result = ocr.ocr(str(image_path), cls=True)

        lines: List[str] = []
        confidences: List[float] = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = float(line[1][1])
                lines.append(text)
                confidences.append(conf)

        full_text = "\n".join(lines)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return OCRResult(text=full_text, confidence=avg_conf, engine="paddleocr")

    # ── EasyOCR (class-cached) ────────────────────────────────
    @classmethod
    def _get_easy_reader(cls):
        """Return a cached EasyOCR reader instance (thread-safe init)."""
        if cls._easy_reader is not None:
            return cls._easy_reader
        with cls._cache_lock:
            if cls._easy_reader is None:
                import easyocr
                cls._easy_reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
                logger.info("EasyOCR reader cached")
        return cls._easy_reader

    def _run_easyocr(self, image_path: Path) -> OCRResult:
        reader = self._get_easy_reader()
        result = reader.readtext(str(image_path), detail=1, paragraph=False)

        lines: List[str] = []
        confidences: List[float] = []
        for (bbox, text, conf) in result:
            lines.append(text)
            confidences.append(conf)

        full_text = "\n".join(lines)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return OCRResult(text=full_text, confidence=avg_conf, engine="easyocr")

    # ── Tesseract ─────────────────────────────────────────────
    def _run_tesseract(self, image_path: Path) -> OCRResult:
        import pytesseract
        from PIL import Image

        img = Image.open(str(image_path))
        data = pytesseract.image_to_data(img, lang="vie+eng",
                                         output_type=pytesseract.Output.DICT)
        words: List[str] = []
        confs: List[float] = []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 0 and word.strip():
                words.append(word)
                confs.append(conf / 100.0)

        full_text = " ".join(words)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return OCRResult(text=full_text, confidence=avg_conf, engine="tesseract")

    # ── Gemini Vision ─────────────────────────────────────────
    def _run_gemini_vision(self, image_path: Path) -> OCRResult:
        import base64
        import google.generativeai as genai

        genai.configure(api_key=self._gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        with open(str(image_path), "rb") as f:
            image_bytes = f.read()

        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp",
                    ".bmp": "image/bmp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        prompt = (
            "Đây là ảnh của một bảng báo giá / bảng giá. "
            "Hãy trích xuất TOÀN BỘ văn bản trong ảnh, giữ nguyên cấu trúc bảng "
            "bằng cách dùng tab hoặc | để phân cách cột. "
            "Trả về text thuần túy, không giải thích thêm."
        )

        response = model.generate_content([
            {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()},
            prompt
        ])

        text = response.text.strip() if response.text else ""
        conf = 0.85 if len(text) > 20 else 0.3
        return OCRResult(text=text, confidence=conf, engine="gemini_vision")


# Convenience type alias so type-hints work cleanly
EasyReaderWrapper = "easyocr.Reader"   # actual type is injected at runtime
