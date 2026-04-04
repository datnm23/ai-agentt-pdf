# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Agent PDF Báo Giá — a system that automatically extracts structured price quote data (products, prices, supplier info) from any file format: soft PDFs, scanned PDFs, and photos of quotes. Built with FastAPI + LangChain + Gemini Flash + multi-engine OCR.

## Commands

```bash
# Development
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Docker
docker compose up --build

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Processing Pipeline (async, background tasks)

All pipeline steps run via `background_tasks.add_task(process_document, ...)` — the HTTP response returns immediately with `job_id`. `process_document` in `app/services/job_service.py` is the orchestrator. It owns its own `asyncio.Semaphore` to enforce `MAX_CONCURRENT_JOBS`. DB writes happen inside the semaphore context.

The pipeline branches by document type detected at Stage 1:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Stage 1 — Detect (0-10%)                             │
│  detect_document_type() → SOFT_PDF | SCANNED_PDF | IMAGE_*                 │
└────────────┬──────────────────────┬──────────────────────┬──────────────────┘
             │                      │                      │
     SOFT_PDF (native)     SCANNED_PDF (images)     IMAGE_* (photo)
             │                      │                      │
   Stage 2a (10-50%)       Stage 2b (12-45%)       Stage 2c (15-40%)
   pdfplumber text          pdf2image →             preprocess →
   + table extract          Gemini Vision           OCR engines
   garbled? → SCANNED       per page                low conf? → Vision
             │                      │                      │
             └──────────────────────┴──────────────────────┘
                                    │
                          Stage 3 — AI Extract (50-90%)
                          ExtractionAgent.extract_from_text()
                          or result already saved (Vision path)
                                    │
                          Stage 4 — Save (90-100%)
                          results/{job_id}.json + DB
```

**SOFT_PDF path** (`app/services/job_service.py:110-135`):
1. `PDFParser.extract_text()` → pdfplumber raw text
2. `_is_garbled_text()` check — if garbled, re-route to SCANNED_PDF path
3. `PDFParser.extract_tables()` → `tables_to_text()` (Markdown format) prepended to text
4. `ExtractionAgent.extract_from_text()` → single Gemini Flash call (no chunking for files ≤400k chars)

**SCANNED_PDF path** (`app/services/job_service.py:137-194`) — Vision-First:
1. `PDFParser.pdf_to_images()` → JPEG per page (200 DPI)
2. `ExtractionAgent.extract_from_image()` called per page → `PriceDocument` per page
3. **Per-page `_inherit_sparse_fields()`** applied before appending — prevents cross-page field contamination (nhom_sp, dvt, vat_pct from page N cannot leak into page N+1)
4. `_merge_documents()` deduplicates by composite key `(ten_sp, don_gia)`
5. `_post_process(doc, skip_field_inheritance=True)` — skips inheritance since already done per-page
6. Fallback: if Gemini returns no items, falls through to text OCR pipeline

**IMAGE path** (`app/services/job_service.py:196-231`):
1. `PreprocessingPipeline.process()` — denoise / deskew / threshold
2. `OCREngineSelector.extract_text()` → PaddleOCR → EasyOCR → Tesseract cascade
3. If OCR confidence < 0.5 → switch to `ExtractionAgent.extract_from_image()` (Gemini Vision), return early
4. Otherwise `extract_from_text()` on OCR output

### Singleton Pattern for Expensive Components
All heavy ML components are lazily initialized with double-check locking:

| Component | Getter | Class-level cache |
|---|---|---|
| PaddleOCR | `_get_paddle_ocr()` | `OCREngineSelector._paddle_ocr` |
| EasyOCR Reader | `_get_easy_reader()` | `OCREngineSelector._easy_reader` |
| PreprocessingPipeline | `_get_preprocessor()` | `_preprocessor` (module global) |
| OCREngineSelector | `_get_ocr_selector()` | `_ocr_selector` (module global) |
| ExtractionAgent | `_get_extraction_agent()` | `_extraction_agent` (module global) |
| PDFParser | `_get_pdf_parser()` | `_pdf_parser` (module global) |

`threading.Lock` is used for the class-level caches in `engine_selector.py`; module-level singletons in `job_service.py` use a shared `_singleton_lock`.

### Database Sessions
`update_job()`, `create_job()`, `get_job()`, `list_jobs()`, `delete_job()` all own their own `AsyncSessionLocal` context internally — callers never pass a session object. Do not add a `session` parameter to these functions.

### OCR Engine Selection
`OCREngineSelector` tries engines in priority order: PaddleOCR → EasyOCR → Tesseract → Gemini Vision. When confidence is below threshold, it falls through to the next engine. Results are returned as `OCRResult(text, confidence, engine)`.

Engine name → `ExtractionMethod` value conversion uses `ENGINE_METHOD_MAP` dict (defined in `engine_selector.py`), not `ExtractionMethod(f"...")`. Always import and use `engine_to_method()` from `engine_selector.py`.

### Gemini Extraction & Chunking
`ExtractionAgent.extract_from_text()` handles texts longer than `MAX_CHARS_PER_CHUNK` or `MAX_ROWS_PER_CHUNK` by splitting at line boundaries with 3-line overlap, calling the LLM per chunk, then deduplicating by composite key `(ten_sp, don_gia)`. Use `_split_text()`, `_extract_chunk()`, `_merge_documents()` — do not truncate text before sending to the LLM.

Chunking is **row-count-based** (primary) with a character cap (secondary):
- `MAX_ROWS_PER_CHUNK = 50`: primary limit on Markdown table data rows per chunk. Gemini 2.5 Flash silently skips middle rows when given 90+ rows in one call (dense catalogs with many size variants reach this easily even at 15k chars).
- `MAX_CHARS_PER_CHUNK = 15_000`: secondary cap for non-table text (plain OCR output without `|` delimiters).

When chunking does occur, `_detect_table_header()` finds the column header row in chunk 1 (line containing ≥3 of: stt/mã/tên/đvt/đơn giá/…) and prepends it as `[HEADER BẢNG TỪ TRANG TRƯỚC]: <header>` to chunks 2..N so the LLM always knows what each column means.

Retry with exponential backoff: 3 attempts, starting at 2s, doubling each time. After all retries fail, the exception propagates to the job's exception handler (job status set to `failed`), not silently swallowed.

### Post-Processing & Field Quality (`_post_process`)
Called once on the final merged document. Steps in order:
1. **Renumber** `stt` globally 1..N
2. **Inherit sparse fields** — `_inherit_sparse_fields()` propagates `nhom_sp`, `dvt`, `vat_pct` from the nearest previous item that has a value. Skipped (`skip_field_inheritance=True`) when already applied per-page in the SCANNED_PDF path.
3. **price_list cleanup** — clear `so_luong`, `thanh_tien`, and document-level totals (meaningless for a catalog)
4. **Auto-calculate** `thanh_tien = so_luong × don_gia` when missing (quote/invoice)
5. **Infer `dvt`** from product name keywords when missing (e.g. "cáp" → "m")
6. **VAT totals** — compute `thue_vat_tien` and `tong_sau_vat` from `tong_chua_vat × thue_vat_pct` when not already set
7. **Cross-validation** (quote/invoice only) — if `abs(so_luong × don_gia − thanh_tien) / thanh_tien > 2%`, lower `confidence` to ≤ 0.5 and append `[⚠️ Kiểm tra: SL×Đơn giá=X ≠ Thành tiền=Y]` to `ghi_chu`

### Table Text Format
`PDFParser.tables_to_text()` renders DataFrames as **Markdown tables** (`df.to_markdown(index=False)`, requires `tabulate`). This gives the LLM proper `|` delimiters instead of terminal-aligned whitespace, improving column recognition. Falls back to `df.to_string()` if tabulate is unavailable.

### Frontend Real-time Updates
The server supports three status delivery modes:
- **WebSocket** (`/api/ws/{job_id}`) — primary, pushed immediately on state change
- **SSE** (`/api/sse/{job_id}`) — streaming fallback
- **Polling** (`GET /api/jobs/{id}/status`) — client falls back when WebSocket fails

The frontend tries WebSocket first (detected via `ws_url` in upload response), auto-falls back to polling on disconnect.

### API Authentication
API key auth is enabled when `API_KEY` env var is set. Validate via `X-API-Key` header or `?api_key=` query param. If `API_KEY` is blank/empty, auth is disabled — the `verify_api_key` dependency is a no-op. Rate limiting (30 req/min per IP) is always active.

### Docker
The `Dockerfile` is multi-stage: builder stage installs Python deps, runtime stage copies them. OCR tools (Tesseract with Vietnamese + English language data, Poppler) are installed at the OS level. The container runs as non-root `appuser`. `docker-compose.yml` caps memory at 4G, persists `uploads/` and `results/` as named volumes.

## Key Conventions

- **Environment variables**: All config via `.env` (use `.env.example` as reference). Key vars: `GOOGLE_API_KEY`, `API_KEY`, `MAX_CONCURRENT_JOBS`, `ALLOWED_ORIGINS`, `MAX_FILE_SIZE_MB`.
- **Path resolution**: All paths use `Path(__file__).resolve()` relative to the file, never `os.getcwd()`.
- **Result persistence**: Results saved to both `results/{job_id}.json` (file) and `result_json` column in SQLite. The JSON file is the canonical source; the DB column is for metadata lookup.
- **Pydantic models**: Use `model_dump()` for serialization (not `.dict()`), `model_validate()` for parsing.
- **Vietnamese price normalization**: `_safe_float()` in `extraction_agent.py` strips VND/đ/$/USD/EUR suffixes and normalizes decimal separators before converting to float.
