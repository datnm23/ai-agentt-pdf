"""FastAPI routes — all REST endpoints for the PDF processing agent."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import (
    AsyncSessionLocal, create_job, delete_job, get_job, list_jobs, get_session
)
from app.models.schemas import HealthResponse, JobStatus
from app.services.job_service import (
    create_job_id, process_document, UPLOAD_DIR,
    _get_ocr_selector,   # cached singleton — no re-init on every health ping
)
from app.services.export_service import export_csv, export_excel, load_result

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp", ".heic"}
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ── Rate limiting ────────────────────────────────────────────────
# Sliding-window rate limiter: client_ip → list of request timestamps
_RL_WINDOW_SECS = 60
_RL_MAX_REQUESTS = 30
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window = _rate_limit_store[client_ip]
    # Expire old timestamps
    window[:] = [t for t in window if now - t < _RL_WINDOW_SECS]
    if len(window) >= _RL_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down."
        )
    window.append(now)


# ── API Key authentication ───────────────────────────────────────
_API_KEY = os.getenv("API_KEY", "")


def verify_api_key(request: Request) -> None:
    """Validate API key from X-API-Key header (or ?api_key= query param)."""
    if not _API_KEY:
        return   # auth disabled when API_KEY is not set

    provided = request.headers.get("x-api-key") or request.query_params.get("api_key", "")
    if not provided:
        raise HTTPException(status_code=401, detail="Missing API key")
    key_hash = hashlib.sha256(provided.encode()).hexdigest()
    if key_hash != _API_KEY and provided != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── WebSocket connection manager ─────────────────────────────────
class ConnectionManager:
    """Maps job_id → set of WebSocket clients waiting for updates."""

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        async with self._lock:
            self._connections[job_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, job_id: str):
        async with self._lock:
            self._connections[job_id].discard(websocket)
            if not self._connections[job_id]:
                del self._connections[job_id]

    async def send_update(self, job_id: str, data: dict):
        """Broadcast update to all clients watching this job."""
        payload = f"data: {json.dumps(data, default=str)}\n\n"
        dead = set()
        async with self._lock:
            clients = list(self._connections.get(job_id, []))
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[job_id].discard(ws)


_manager = ConnectionManager()


# ── Health ───────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Reuse the cached OCREngineSelector singleton."""
    selector = _get_ocr_selector()
    return HealthResponse(status="ok", version="1.0.0", ocr_engines=selector._available)


# ── Upload ──────────────────────────────────────────────────────
@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    _=Depends(verify_api_key),
):
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không được hỗ trợ: {suffix}. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413,
                            detail=f"File quá lớn: {size_mb:.1f}MB (tối đa {MAX_FILE_MB}MB)")

    # Save uploaded file
    job_id = create_job_id()
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / file.filename.replace(" ", "_")

    with open(str(file_path), "wb") as f:
        f.write(content)

    # Create DB record
    async with AsyncSessionLocal() as session:
        await create_job(session, job_id, file.filename, str(file_path))

    # Start background processing (also push updates via WebSocket)
    background_tasks.add_task(_run_with_ws, job_id, file_path, file.filename)
    logger.info(f"Job {job_id} queued for {file.filename} ({size_mb:.1f}MB)")

    return {
        "job_id": job_id,
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "status": "pending",
        "message": "Đã nhận file, đang bắt đầu xử lý...",
        "ws_url": f"/api/ws/{job_id}",   # client can opt in to WebSocket
    }


async def _run_with_ws(job_id: str, file_path: Path, filename: str):
    """Wrapper that pushes WebSocket updates after each pipeline step."""
    try:
        await process_document(job_id, file_path, filename)
    finally:
        # Notify all WS clients the job is done/failed
        async with AsyncSessionLocal() as session:
            record = await get_job(session, job_id)
        if record:
            await _manager.send_update(job_id, {
                "job_id": job_id,
                "status": record.status,
                "current_step": record.current_step,
                "progress_pct": record.progress_pct,
            })


# ── WebSocket (real-time status) ─────────────────────────────────
@router.websocket("/ws/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    await _manager.connect(websocket, job_id)
    try:
        # Send current state immediately on connect
        async with AsyncSessionLocal() as session:
            record = await get_job(session, job_id)
        if record:
            await websocket.send_json({
                "job_id": job_id,
                "status": record.status,
                "current_step": record.current_step,
                "progress_pct": record.progress_pct,
            })
        # Keep alive — client just listens
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                # Accept ping/pong or heartbeat
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text(": keepalive\n\n")
    except WebSocketDisconnect:
        pass
    finally:
        await _manager.disconnect(websocket, job_id)


# ── SSE (Server-Sent Events — alternative to WebSocket) ──────────
@router.get("/sse/{job_id}")
async def sse_status(job_id: str):
    """SSE stream for real-time job status updates."""
    async def event_generator():
        last_status = None
        while True:
            async with AsyncSessionLocal() as session:
                record = await get_job(session, job_id)

            if record:
                data = {
                    "job_id": job_id,
                    "status": record.status,
                    "current_step": record.current_step,
                    "progress_pct": record.progress_pct,
                }
                # Only send on change
                if data["status"] != last_status:
                    yield f"data: {json.dumps(data, default=str)}\n\n"
                    last_status = data["status"]

                if record.status in (JobStatus.DONE.value, JobStatus.FAILED.value):
                    break

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Job Status (polling fallback) ────────────────────────────────
@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str, request: Request, _=Depends(verify_api_key)):
    _check_rate_limit(request.client.host if request.client else "unknown")
    async with AsyncSessionLocal() as session:
        record = await get_job(session, job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job {job_id} không tồn tại")

    return {
        "job_id": record.job_id,
        "status": record.status,
        "current_step": record.current_step,
        "progress_pct": record.progress_pct,
        "document_type": record.document_type,
        "extraction_method": record.extraction_method,
        "error": record.error_message,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }


# ── Results ──────────────────────────────────────────────────────
@router.get("/results/{job_id}")
async def get_results(job_id: str, request: Request, _=Depends(verify_api_key)):
    _check_rate_limit(request.client.host if request.client else "unknown")
    async with AsyncSessionLocal() as session:
        record = await get_job(session, job_id)

    if not record:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    if record.status != JobStatus.DONE.value:
        raise HTTPException(status_code=202, detail=f"Job chưa hoàn thành: {record.status}")

    doc = load_result(job_id)
    if not doc:
        raise HTTPException(status_code=500, detail="Không tìm thấy file kết quả")

    return {
        "job_id": job_id,
        "filename": record.filename,
        "document_type": record.document_type,
        "extraction_method": record.extraction_method,
        "overall_confidence": record.overall_confidence,
        "result": doc.model_dump(),
    }


# ── Export ──────────────────────────────────────────────────────
@router.get("/export/{job_id}")
async def export_results(job_id: str, format: str = "excel", _=Depends(verify_api_key)):
    if format == "excel":
        path = export_excel(job_id)
        if not path:
            raise HTTPException(status_code=404, detail="Không có dữ liệu để export")
        return FileResponse(
            str(path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"baogia_{job_id}.xlsx",
        )
    elif format == "csv":
        path = export_csv(job_id)
        if not path:
            raise HTTPException(status_code=404, detail="Không có dữ liệu để export")
        return FileResponse(str(path), media_type="text/csv", filename=f"baogia_{job_id}.csv")
    elif format == "json":
        doc = load_result(job_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        return JSONResponse(doc.model_dump())
    else:
        raise HTTPException(status_code=400, detail="Format không hợp lệ. Dùng: excel | csv | json")


# ── History ─────────────────────────────────────────────────────
@router.get("/history")
async def get_history(limit: int = 20, _=Depends(verify_api_key)):
    async with AsyncSessionLocal() as session:
        records = await list_jobs(session, limit=limit)

    return [
        {
            "job_id": r.job_id,
            "filename": r.filename,
            "status": r.status,
            "document_type": r.document_type,
            "extraction_method": r.extraction_method,
            "overall_confidence": r.overall_confidence,
            "progress_pct": r.progress_pct,
            "current_step": r.current_step,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in records
    ]


# ── Delete ──────────────────────────────────────────────────────
@router.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str, _=Depends(verify_api_key)):
    async with AsyncSessionLocal() as session:
        record = await get_job(session, job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job không tồn tại")
        await delete_job(session, job_id)

    # Cleanup files
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(str(job_dir))

    results_file = Path(__file__).resolve().parent.parent.parent / "results" / f"{job_id}.json"
    if results_file.exists():
        results_file.unlink()

    return {"message": f"Job {job_id} đã được xóa"}
