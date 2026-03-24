"""Export service — Excel, CSV, JSON export from processed results."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from app.models.schemas import PriceDocument, PriceItem

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def load_result(job_id: str) -> Optional[PriceDocument]:
    result_file = RESULTS_DIR / f"{job_id}.json"
    if not result_file.exists():
        return None
    with open(str(result_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    return PriceDocument(**data)


def export_excel(job_id: str) -> Optional[Path]:
    doc = load_result(job_id)
    if not doc or not doc.items:
        return None

    output_path = RESULTS_DIR / f"{job_id}_export.xlsx"
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:

        # ── Sheet 1: Items ─────────────────────────────────────
        rows = []
        for i, item in enumerate(doc.items, 1):
            rows.append({
                "STT": item.stt or i,
                "Mã SP": item.ma_sp or "",
                "Tên sản phẩm / Dịch vụ": item.ten_sp,
                "ĐVT": item.dvt or "",
                "Số lượng": item.so_luong,
                "Đơn giá": item.don_gia,
                "Chiết khấu %": item.chiet_khau_pct,
                "Thành tiền": item.thanh_tien,
                "Ghi chú": item.ghi_chu or "",
                "Độ tin cậy AI": f"{item.confidence:.0%}",
            })

        df_items = pd.DataFrame(rows)
        df_items.to_excel(writer, sheet_name="Bảng Giá", index=False)

        # Format worksheet
        ws = writer.sheets["Bảng Giá"]
        _format_excel_sheet(ws, df_items)

        # ── Sheet 2: Summary ───────────────────────────────────
        summary_data = {
            "Thông tin": [
                "Nhà cung cấp", "Số báo giá", "Ngày báo giá",
                "Khách hàng", "Đơn vị tiền", "Số sản phẩm",
                "Tổng chưa VAT", f"Thuế VAT ({doc.thue_vat_pct or 10}%)",
                "Tổng sau VAT", "Điều kiện thanh toán", "Thời gian giao hàng",
                "Bảo hành",
            ],
            "Giá trị": [
                doc.nha_cung_cap or "", doc.so_bao_gia or "",
                doc.ngay_bao_gia or "", doc.khach_hang or "",
                doc.don_vi_tien, len(doc.items),
                _fmt_money(doc.tong_chua_vat, doc.don_vi_tien),
                _fmt_money(doc.thue_vat_tien, doc.don_vi_tien),
                _fmt_money(doc.tong_sau_vat, doc.don_vi_tien),
                doc.dieu_kien_thanh_toan or "", doc.thoi_gian_giao_hang or "",
                doc.bao_hanh or "",
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Tóm Tắt", index=False)

    logger.info(f"Excel exported: {output_path.name}")
    return output_path


def export_csv(job_id: str) -> Optional[Path]:
    doc = load_result(job_id)
    if not doc or not doc.items:
        return None

    output_path = RESULTS_DIR / f"{job_id}_export.csv"
    rows = []
    for i, item in enumerate(doc.items, 1):
        rows.append({
            "stt": item.stt or i,
            "ma_sp": item.ma_sp or "",
            "ten_sp": item.ten_sp,
            "dvt": item.dvt or "",
            "so_luong": item.so_luong,
            "don_gia": item.don_gia,
            "chiet_khau_pct": item.chiet_khau_pct,
            "thanh_tien": item.thanh_tien,
            "ghi_chu": item.ghi_chu or "",
            "confidence": item.confidence,
        })
    pd.DataFrame(rows).to_csv(str(output_path), index=False, encoding="utf-8-sig")
    return output_path


def _fmt_money(val: Optional[float], currency: str = "VND") -> str:
    if val is None:
        return ""
    if currency == "VND":
        return f"{val:,.0f} đ"
    return f"{val:,.2f} {currency}"


def _format_excel_sheet(ws, df: pd.DataFrame):
    """Apply basic Excel formatting."""
    try:
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        # Header row styling
        header_fill = PatternFill(start_color="1a56db", end_color="1a56db", fill_type="solid")
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-width columns
        for col_idx, col in enumerate(df.columns, 1):
            max_len = max(len(str(col)), df[col].astype(str).str.len().max() or 10)
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 50)

        # Zebra stripes
        light_fill = PatternFill(start_color="EBF3FA", end_color="EBF3FA", fill_type="solid")
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            if row[0].row % 2 == 0:
                for cell in row:
                    cell.fill = light_fill
    except Exception as e:
        logger.warning(f"Excel formatting failed: {e}")
