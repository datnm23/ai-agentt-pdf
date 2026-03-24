"""Job orchestration service — coordinates the full processing pipeline."""
from __future__ import annotations
import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.db.database import update_job
from app.models.schemas import DocumentType, ExtractionMethod, JobStatus, PriceDocument
from app.input.detector import detect_document_type
from app.preprocessing.pipeline import PreprocessingPipeline
from app.ocr.engine_selector import OCREngineSelector, engine_to_method
from app.agents.extraction_agent import ExtractionAgent
from app.input.pdf_parser import PDFParser

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Concurrency limiter ──────────────────────────────────────────
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
_job_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
logger.info(f"Concurrency limit: {_MAX_CONCURRENT} simultaneous jobs")

# ── Thread-safe lazy singletons ──────────────────────────────────
_preprocessor: Optional[PreprocessingPipeline] = None
_ocr_selector: Optional[OCREngineSelector] = None
_extraction_agent: Optional[ExtractionAgent] = None
_pdf_parser: Optional[PDFParser] = None
_singleton_lock = threading.Lock()


def _get_preprocessor() -> PreprocessingPipeline:
    global _preprocessor
    if _preprocessor is None:
        with _singleton_lock:
            if _preprocessor is None:   # double-check
                _preprocessor = PreprocessingPipeline()
    return _preprocessor


def _get_ocr_selector() -> OCREngineSelector:
    global _ocr_selector
    if _ocr_selector is None:
        with _singleton_lock:
            if _ocr_selector is None:
                api_key = os.getenv("GOOGLE_API_KEY", "")
                _ocr_selector = OCREngineSelector(google_api_key=api_key)
    return _ocr_selector


def _get_extraction_agent() -> ExtractionAgent:
    global _extraction_agent
    if _extraction_agent is None:
        with _singleton_lock:
            if _extraction_agent is None:
                api_key = os.getenv("GOOGLE_API_KEY", "")
                _extraction_agent = ExtractionAgent(google_api_key=api_key)
    return _extraction_agent


def _get_pdf_parser() -> PDFParser:
    global _pdf_parser
    if _pdf_parser is None:
        with _singleton_lock:
            if _pdf_parser is None:
                _pdf_parser = PDFParser()
    return _pdf_parser


async def _update(job_id: str, status: str, step: str, progress: int, **kwargs):
    """Helper to update job status in DB."""
    await update_job(job_id,
                     status=status,
                     current_step=step,
                     progress_pct=progress,
                     **kwargs)


async def process_document(job_id: str, file_path: Path, filename: str):
    """
    Full processing pipeline — runs in background task.
    Steps: Detect → Preprocess → OCR/Parse → AI Extract → Save
    """
    async with _job_semaphore:   # ← enforces MAX_CONCURRENT_JOBS
        try:
            # ── Step 1: Detect document type ──────────────────
            await _update(job_id, JobStatus.DETECTING, "🔍 Phân loại tài liệu...", 10)
            doc_type, type_confidence = await asyncio.get_event_loop().run_in_executor(
                None, detect_document_type, file_path
            )
            logger.info(f"[{job_id}] Detected: {doc_type} (conf={type_confidence:.2f})")
            await update_job(job_id, document_type=doc_type.value)

            # ── Step 2: Extract text ───────────────────────────
            extracted_text = ""
            extraction_method = ExtractionMethod.PDFPLUMBER
            preview_path: Optional[Path] = None

            if doc_type == DocumentType.SOFT_PDF:
                # Native PDF — use pdfplumber + camelot
                await _update(job_id, JobStatus.OCR, "📄 Đọc text từ PDF...", 30)
                parser = _get_pdf_parser()
                extracted_text = await asyncio.get_event_loop().run_in_executor(
                    None, parser.extract_text, file_path
                )
                tables = await asyncio.get_event_loop().run_in_executor(
                    None, parser.extract_tables, file_path
                )
                if tables:
                    table_text = parser.tables_to_text(tables)
                    extracted_text = f"{table_text}\n\n{extracted_text}"
                extraction_method = ExtractionMethod.PDFPLUMBER

            elif doc_type == DocumentType.SCANNED_PDF:
                # Scanned PDF — convert to images then OCR
                await _update(job_id, JobStatus.PREPROCESSING,
                              "🔄 Chuyển PDF sang ảnh...", 25)
                parser = _get_pdf_parser()
                img_dir = UPLOAD_DIR / job_id / "pages"
                image_paths = await asyncio.get_event_loop().run_in_executor(
                    None, parser.pdf_to_images, file_path, img_dir
                )
                if image_paths:
                    preview_path = image_paths[0]
                    await _update(job_id, JobStatus.PREPROCESSING,
                                  "🖼️ Xử lý ảnh...", 35)
                    preprocessor = _get_preprocessor()
                    processed_texts = []
                    for img_path in image_paths:
                        proc_path = await asyncio.get_event_loop().run_in_executor(
                            None, preprocessor.process, img_path
                        )
                        await _update(job_id, JobStatus.OCR,
                                      f"🔤 OCR trang {img_path.stem}...", 50)
                        ocr = _get_ocr_selector()
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ocr.extract_text, proc_path
                        )
                        processed_texts.append(result.text)
                        # Safe enum mapping — no runtime crash on unknown engine
                        method_str = engine_to_method(result.engine)
                        extraction_method = ExtractionMethod(method_str)
                    extracted_text = "\n\n".join(processed_texts)

            else:
                # Direct image input (JPG, PNG, HEIC, etc.)
                preview_path = file_path
                await _update(job_id, JobStatus.PREPROCESSING,
                              "🖼️ Tiền xử lý ảnh (glare/deskew)...", 25)
                preprocessor = _get_preprocessor()
                proc_dir = UPLOAD_DIR / job_id
                proc_dir.mkdir(parents=True, exist_ok=True)
                processed_image = await asyncio.get_event_loop().run_in_executor(
                    None, preprocessor.process, file_path,
                    proc_dir / f"proc_{file_path.name}"
                )
                await _update(job_id, JobStatus.OCR,
                              "🔤 Nhận dạng văn bản (OCR)...", 50)
                ocr = _get_ocr_selector()
                ocr_result = await asyncio.get_event_loop().run_in_executor(
                    None, ocr.extract_text, processed_image
                )
                extracted_text = ocr_result.text
                # Safe enum mapping
                method_str = engine_to_method(ocr_result.engine)
                extraction_method = ExtractionMethod(method_str)

                # Low confidence → switch to Gemini Vision directly
                if ocr_result.confidence < 0.5:
                    logger.info(
                        f"[{job_id}] Low OCR confidence ({ocr_result.confidence:.2f}), "
                        "switching to Gemini Vision"
                    )
                    await _update(job_id, JobStatus.OCR,
                                  "👁️ Gemini Vision đang phân tích ảnh...", 55)
                    agent = _get_extraction_agent()
                    doc = await asyncio.get_event_loop().run_in_executor(
                        None, agent.extract_from_image, processed_image
                    )
                    extraction_method = ExtractionMethod.GEMINI_VISION
                    await _save_result(job_id, doc, extraction_method,
                                       preview_path, ocr_result.confidence)
                    return

            # ── Step 3: AI Extraction ─────────────────────────
            await _update(job_id, JobStatus.EXTRACTING,
                          "🤖 AI đang phân tích dữ liệu...", 70)
            agent = _get_extraction_agent()
            doc = await asyncio.get_event_loop().run_in_executor(
                None, agent.extract_from_text, extracted_text, filename
            )

            # ── Step 4: Save results ──────────────────────────
            await _save_result(job_id, doc, extraction_method,
                               preview_path, 0.85)

        except Exception as e:
            logger.exception(f"[{job_id}] Processing failed: {e}")
            await update_job(job_id,
                             status=JobStatus.FAILED.value,
                             error_message=str(e),
                             current_step=f"❌ Lỗi: {str(e)[:200]}",
                             progress_pct=0)


async def _save_result(job_id: str, doc: PriceDocument,
                       method: ExtractionMethod, preview_path: Optional[Path],
                       confidence: float):
    """Save extraction result to DB and JSON file."""
    result_file = RESULTS_DIR / f"{job_id}.json"
    doc_dict = doc.model_dump()

    # Calculate overall confidence from items
    if doc.items:
        avg_item_conf = sum(item.confidence for item in doc.items) / len(doc.items)
        overall_conf = (avg_item_conf + confidence) / 2
    else:
        overall_conf = confidence

    with open(str(result_file), "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    await update_job(job_id,
                     status=JobStatus.DONE.value,
                     extraction_method=method.value,
                     overall_confidence=overall_conf,
                     progress_pct=100,
                     current_step=f"✅ Hoàn thành — {len(doc.items)} sản phẩm",
                     completed_at=datetime.utcnow(),
                     result_json=json.dumps(doc_dict, ensure_ascii=False),
                     preview_image_path=str(preview_path) if preview_path else None)
    logger.info(f"[{job_id}] Done. {len(doc.items)} items, confidence={overall_conf:.2f}")


def create_job_id() -> str:
    return str(uuid.uuid4())[:12].replace("-", "")
