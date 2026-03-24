"""Database layer — SQLAlchemy async with SQLite."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Column, String, Float, Integer, Text, DateTime, select, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# ── DB path ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "results" / "jobs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")
    document_type = Column(String, nullable=True)
    extraction_method = Column(String, nullable=True)
    overall_confidence = Column(Float, default=0.0)
    progress_pct = Column(Integer, default=0)
    current_step = Column(String, default="Đang chờ...")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_json = Column(Text, nullable=True)        # JSON of PriceDocument
    preview_image_path = Column(String, nullable=True)
    file_path = Column(String, nullable=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


# ── CRUD helpers ──────────────────────────────────────────────

async def create_job(session: AsyncSession, job_id: str, filename: str, file_path: str) -> JobRecord:
    record = JobRecord(job_id=job_id, filename=filename, file_path=file_path)
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def update_job(job_id: str, **kwargs: Any) -> Optional[JobRecord]:
    """Update a job record. Owns its DB session (callers no longer need to pass one)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(JobRecord).where(JobRecord.job_id == job_id))
        record = result.scalar_one_or_none()
        if record:
            for key, value in kwargs.items():
                setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
        return record


async def get_job(session: AsyncSession, job_id: str) -> Optional[JobRecord]:
    result = await session.execute(select(JobRecord).where(JobRecord.job_id == job_id))
    return result.scalar_one_or_none()


async def list_jobs(session: AsyncSession, limit: int = 50) -> list[JobRecord]:
    result = await session.execute(
        select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def delete_job(session: AsyncSession, job_id: str) -> bool:
    await session.execute(delete(JobRecord).where(JobRecord.job_id == job_id))
    await session.commit()
    return True
