"""Pydantic schemas for price document data models."""
from __future__ import annotations
from typing import List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    SOFT_PDF = "soft_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE_SCAN = "image_scan"
    IMAGE_PHOTO = "image_photo"
    IMAGE_PHOTO_GLARE = "image_photo_glare"
    IMAGE_SKEWED = "image_skewed"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    PDFPLUMBER = "pdfplumber"
    CAMELOT = "camelot"
    OCR_PADDLE = "ocr_paddleocr"
    OCR_EASY = "ocr_easyocr"
    OCR_TESSERACT = "ocr_tesseract"
    GEMINI_VISION = "gemini_vision"


class JobStatus(str, Enum):
    PENDING = "pending"
    DETECTING = "detecting"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    EXTRACTING = "extracting"
    DONE = "done"
    FAILED = "failed"


class PriceItem(BaseModel):
    """Một dòng sản phẩm trong bảng giá."""
    stt: Optional[int] = Field(None, description="Số thứ tự")
    ma_sp: Optional[str] = Field(None, description="Mã sản phẩm")
    ten_sp: str = Field(description="Tên sản phẩm / dịch vụ")
    dvt: Optional[str] = Field(None, description="Đơn vị tính (cái, m, kg...)")
    so_luong: Optional[float] = Field(None, description="Số lượng")
    don_gia: Optional[float] = Field(None, description="Đơn giá (VND hoặc ngoại tệ)")
    thanh_tien: Optional[float] = Field(None, description="Thành tiền = SL × Đơn giá")
    chiet_khau_pct: Optional[float] = Field(None, description="Chiết khấu %")
    chiet_khau_tien: Optional[float] = Field(None, description="Chiết khấu tiền")
    ghi_chu: Optional[str] = Field(None, description="Ghi chú, xuất xứ, bảo hành...")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Độ tin cậy 0-1")


class PriceDocument(BaseModel):
    """Toàn bộ thông tin một bảng báo giá."""
    nha_cung_cap: Optional[str] = Field(None, description="Tên công ty / nhà cung cấp")
    dia_chi: Optional[str] = None
    dien_thoai: Optional[str] = None
    email: Optional[str] = None
    so_bao_gia: Optional[str] = Field(None, description="Số báo giá / PO")
    ngay_bao_gia: Optional[str] = Field(None, description="Ngày lập báo giá")
    ngay_het_han: Optional[str] = Field(None, description="Ngày hết hạn báo giá")
    khach_hang: Optional[str] = None
    don_vi_tien: str = Field(default="VND", description="Đơn vị tiền tệ")
    items: List[PriceItem] = Field(default_factory=list)
    tong_chua_vat: Optional[float] = None
    thue_vat_pct: Optional[float] = Field(None, description="% VAT (thường 10)")
    thue_vat_tien: Optional[float] = None
    tong_sau_vat: Optional[float] = None
    dieu_kien_thanh_toan: Optional[str] = None
    thoi_gian_giao_hang: Optional[str] = None
    bao_hanh: Optional[str] = None
    ghi_chu_chung: Optional[str] = None


class ProcessingJob(BaseModel):
    """Trạng thái của một job xử lý."""
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    document_type: Optional[DocumentType] = None
    extraction_method: Optional[ExtractionMethod] = None
    overall_confidence: float = 0.0
    progress_pct: int = 0
    current_step: str = "Đang chờ xử lý..."
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[PriceDocument] = None
    preview_image_path: Optional[str] = None


class ExportRequest(BaseModel):
    job_id: str
    format: str = Field(default="excel", pattern="^(excel|csv|json)$")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    ocr_engines: List[str] = []
