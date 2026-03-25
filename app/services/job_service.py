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
from app.models.schemas import DocumentType, ExtractionMethod, JobStatus, PriceDocument, PriceItem
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
    Progress stages:
      Detect (0-10%) → [per-path extract] → AI extract (50-90%) → Save (90-100%)
    """
    async with _job_semaphore:   # ← enforces MAX_CONCURRENT_JOBS
        try:
            # ── Stage 1: Detect (0-10%) ───────────────────────
            await _update(job_id, JobStatus.DETECTING, "🔍 Đang phân loại tài liệu...", 5)
            doc_type, type_confidence = await asyncio.get_event_loop().run_in_executor(
                None, detect_document_type, file_path
            )
            logger.info(f"[{job_id}] Detected: {doc_type} (conf={type_confidence:.2f})")
            await update_job(job_id, document_type=doc_type.value)

            extracted_text = ""
            extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER
            preview_path: Optional[Path] = None

            # ── Stage 2: Extract (10-50%) ──────────────────────
            if doc_type == DocumentType.SOFT_PDF:
                # Native PDF: pdfplumber text → check garbled → tables
                await _update(job_id, JobStatus.OCR,
                              "📄 Đang đọc text từ PDF...", 15)
                parser = _get_pdf_parser()
                extracted_text = await asyncio.get_event_loop().run_in_executor(
                    None, parser.extract_text, file_path
                )

                if _is_garbled_text(extracted_text):
                    logger.warning(f"[{job_id}] pdfplumber garbled, switching to Gemini Vision")
                    doc_type = DocumentType.SCANNED_PDF
                    extracted_text = ""

            if doc_type == DocumentType.SOFT_PDF:
                # Clean native PDF: extract tables
                await _update(job_id, JobStatus.OCR,
                              "📊 Đang trích xuất bảng...", 30)
                parser = _get_pdf_parser()
                tables = await asyncio.get_event_loop().run_in_executor(
                    None, parser.extract_tables, file_path
                )
                if tables:
                    table_text = parser.tables_to_text(tables)
                    extracted_text = f"{table_text}\n\n{extracted_text}"
                extraction_method = ExtractionMethod.PDFPLUMBER

            elif doc_type == DocumentType.SCANNED_PDF:
                # Scanned PDF → images → Gemini Vision per page
                await _update(job_id, JobStatus.PREPROCESSING,
                              "🔄 Đang chuyển PDF sang ảnh...", 12)
                parser = _get_pdf_parser()
                img_dir = UPLOAD_DIR / job_id / "pages"
                image_paths = await asyncio.get_event_loop().run_in_executor(
                    None, parser.pdf_to_images, file_path, img_dir
                )
                total_pages = len(image_paths)
                logger.info(f"[{job_id}] {total_pages} pages → Gemini Vision")
                preview_path = image_paths[0] if image_paths else None

                if image_paths:
                    agent = _get_extraction_agent()
                    all_docs: list[PriceDocument] = []

                    for idx, img_path in enumerate(image_paths, 1):
                        # OCR progress: 15-45% over all pages (50% budget)
                        page_progress = int(15 + 30 * idx / total_pages)
                        await _update(job_id, JobStatus.OCR,
                                      f"👁️ Đang đọc trang {idx}/{total_pages}...",
                                      page_progress)
                        doc = await asyncio.get_event_loop().run_in_executor(
                            None, agent.extract_from_image, img_path
                        )
                        if doc.items:
                            # Scope inheritance to this page before merging —
                            # prevents nhom_sp/vat_pct from page N contaminating page N+1
                            ExtractionAgent._inherit_sparse_fields(doc)
                            all_docs.append(doc)

                    if all_docs:
                        doc = _merge_documents(all_docs)
                        # skip_field_inheritance=True: already done per-page above
                        doc = ExtractionAgent._post_process(doc, skip_field_inheritance=True)
                        extraction_method = ExtractionMethod.GEMINI_VISION
                        await _save_result(job_id, doc, extraction_method,
                                           preview_path, 0.85)
                        return

                    # Gemini returned nothing — fall back to OCR text extraction
                    logger.warning(f"[{job_id}] Gemini no items, trying text OCR")
                    preprocessor = _get_preprocessor()
                    ocr = _get_ocr_selector()
                    for idx, img_path in enumerate(image_paths, 1):
                        await _update(job_id, JobStatus.OCR,
                                      f"🔤 OCR trang {idx}/{total_pages}...",
                                      int(15 + 30 * idx / total_pages))
                        proc_path = await asyncio.get_event_loop().run_in_executor(
                            None, preprocessor.process, img_path
                        )
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ocr.extract_text, proc_path
                        )
                        if result.text.strip():
                            extracted_text += result.text + "\n\n"
                    extraction_method = ExtractionMethod.GEMINI_VISION

            else:
                # Direct image: preprocess → OCR → optional Gemini Vision
                await _update(job_id, JobStatus.PREPROCESSING,
                              "🖼️ Đang tiền xử lý ảnh...", 15)
                preprocessor = _get_preprocessor()
                proc_dir = UPLOAD_DIR / job_id
                proc_dir.mkdir(parents=True, exist_ok=True)
                processed_image = await asyncio.get_event_loop().run_in_executor(
                    None, preprocessor.process, file_path,
                    proc_dir / f"proc_{file_path.name}"
                )
                preview_path = file_path

                await _update(job_id, JobStatus.OCR,
                              "🔤 Đang nhận dạng văn bản (OCR)...", 30)
                ocr = _get_ocr_selector()
                ocr_result = await asyncio.get_event_loop().run_in_executor(
                    None, ocr.extract_text, processed_image
                )
                method_str = engine_to_method(ocr_result.engine)
                extraction_method = ExtractionMethod(method_str)

                if ocr_result.confidence < 0.5:
                    logger.info(f"[{job_id}] Low OCR conf ({ocr_result.confidence:.2f}), "
                                "switching to Gemini Vision")
                    await _update(job_id, JobStatus.OCR,
                                  "👁️ Đang phân tích bằng Gemini Vision...", 40)
                    agent = _get_extraction_agent()
                    doc = await asyncio.get_event_loop().run_in_executor(
                        None, agent.extract_from_image, processed_image
                    )
                    extraction_method = ExtractionMethod.GEMINI_VISION
                    await _save_result(job_id, doc, extraction_method,
                                       preview_path, ocr_result.confidence)
                    return
                extracted_text = ocr_result.text

            # ── Stage 3: AI extraction (50-90%) ────────────────
            # Progress varies: 50-80% for text extraction, 90% for Gemini Vision
            ai_progress = 60 if extraction_method == ExtractionMethod.PDFPLUMBER else 90
            await _update(job_id, JobStatus.EXTRACTING,
                          "🤖 Đang phân tích dữ liệu bằng AI...", ai_progress)
            agent = _get_extraction_agent()
            doc = await asyncio.get_event_loop().run_in_executor(
                None, agent.extract_from_text, extracted_text, filename
            )
            await _update(job_id, JobStatus.EXTRACTING,
                          "💾 Đang lưu kết quả...", 95)

            # ── Stage 4: Save (95-100%) ───────────────────────
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


def _merge_documents(docs: list[PriceDocument]) -> PriceDocument:
    """Merge items from multiple page-level PriceDocuments into one.

    Deduplicates using a composite key (ten_sp, don_gia) so that the same
    product name appearing on different pages with different prices is NOT
    treated as a duplicate — they are distinct line items.
    """
    if not docs:
        return PriceDocument(don_vi_tien="VND", items=[])

    base = docs[0]
    merged_items: list[PriceItem] = []
    # Composite key: name + price distinguishes same-name cables with different specs
    seen_keys: set[tuple[str, float]] = set()

    for doc in docs:
        for item in doc.items:
            name_key = (item.ten_sp or "").strip().lower()
            price_key = item.don_gia or 0.0
            key = (name_key, round(price_key, 2))
            if name_key and key not in seen_keys:
                seen_keys.add(key)
                merged_items.append(item)
            elif not name_key:
                merged_items.append(item)

    base.items = merged_items
    logger.info(f"Merged {len(docs)} pages → {len(merged_items)} unique items")
    return base


# ── Text quality helpers ────────────────────────────────────────

def _is_garbled_text(text: str) -> bool:
    """Return True if text is likely garbled from font encoding issues.

    Signals of garbled Vietnamese PDF text (custom font glyph mapping failed):
    - Very low / zero Vietnamese diacritical chars despite being a Vietnamese doc
    - Many number-substituted chars (0→o, 1→i, 6→ơ, 3→ê, 8→ư)
    - High ratio of all-uppercase "acronym" words (TCVN, UPVC, IP,HCM)
    """
    if not text or len(text) < 50:
        return True

    # Count Vietnamese diacritical chars
    vi_diacritics = sum(1 for c in text if c in
        "ăâđêôơưáàảạấầẩậắằẳặéèẻẹếềểệíìỉịóòỏọốồổộớờởợúùủụứừửựýỳỷỵĐ")
    diacritic_ratio = vi_diacritics / len(text)

    # Count digit-substituted chars (0,1,3,6,8,4 used as glyph replacements)
    sub_chars = sum(text.count(c) for c in "016383456$@#")
    alpha_chars = sum(1 for c in text if c.isalpha())
    sub_ratio = sub_chars / max(alpha_chars, 1)

    # All-uppercase "acronym" words
    import re
    acronym_words = re.findall(r'\b[A-Z]{3,}\b', text)
    total_words = len(re.findall(r'\b\w+\b', text))
    acronym_ratio = len(acronym_words) / max(total_words, 1)

    # Garbled: almost no diacritics + either digit-substitution OR high acronym ratio
    garbled = (
        diacritic_ratio < 0.005          # virtually no Vietnamese diacritics
        and (
            sub_ratio > 0.05            # significant number-substitution
            or acronym_ratio > 0.12      # many all-caps words
        )
    )

    if garbled:
        logger.warning(
            f"Garbled text detected: diacritics={diacritic_ratio:.4f}, "
            f"sub_ratio={sub_ratio:.3f}, acronym_ratio={acronym_ratio:.2f}, "
            f"acronyms={acronym_words[:8]}"
        )
    return garbled
