"""FastAPI application entry point."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

# Load environment variables
load_dotenv()

from app.db.database import init_db
from app.api.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Allowed origins — comma-separated in env or default to frontend port in dev
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("🚀 Starting AI PDF Agent...")
    await init_db()
    logger.info("✅ Database initialized")
    # Ensure upload/results dirs exist
    (BASE_DIR / "uploads").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    yield
    logger.info("🛑 Shutting down AI PDF Agent...")


app = FastAPI(
    title="AI Agent PDF Báo Giá",
    description="Tự động trích xuất dữ liệu từ mọi loại file báo giá/bảng giá",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — restricted origins (not "*") for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_spa():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def catch_all(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true",
    )
