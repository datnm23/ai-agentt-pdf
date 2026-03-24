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
```
detect (10%) → preprocess (25-35%) → OCR (35-55%) → AI extract (70%) → save (100%)
```
All pipeline steps run via `background_tasks.add_task(process_document, ...)` — the HTTP response returns immediately with `job_id`.

`process_document` in `app/services/job_service.py` is the orchestrator. It owns its own `asyncio.Semaphore` to enforce `MAX_CONCURRENT_JOBS`. DB writes happen inside the semaphore context.

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
`ExtractionAgent.extract_from_text()` handles texts longer than `MAX_CHARS_PER_CHUNK` (8000 chars) by splitting at line boundaries with 3-line overlap, calling the LLM per chunk, then deduplicating by product name. Use `_split_text()`, `_extract_chunk()`, `_merge_documents()` — do not truncate text before sending to the LLM.

Retry with exponential backoff: 3 attempts, starting at 2s, doubling each time. After all retries fail, the exception propagates to the job's exception handler (job status set to `failed`), not silently swallowed.

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
