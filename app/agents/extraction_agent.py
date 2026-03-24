"""AI Extraction Agent — LangChain + Gemini Flash with structured output."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from app.models.schemas import PriceDocument, PriceItem

# ── Constants ────────────────────────────────────────────────────
MAX_CHARS_PER_CHUNK = 8000   # Gemini Flash context window guidance
MAX_RETRIES = 3
BASE_BACKOFF_SECS = 2.0

EXTRACTION_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích bảng giá/báo giá của các doanh nghiệp Việt Nam.
Nhiệm vụ của bạn là trích xuất thông tin từ văn bản OCR của một bảng báo giá.

QUY TẮC QUAN TRỌNG:
1. Trích xuất ĐÚNG và ĐẦY ĐỦ tất cả sản phẩm/dịch vụ trong bảng
2. Chuẩn hóa giá tiền: loại bỏ dấu chấm phân cách hàng nghìn, giữ số nguyên
   - "1.200.000" → 1200000  |  "1,200,000" → 1200000  |  "1.2K" → 1200
3. Đơn vị tiền mặc định là VND nếu không ghi rõ
4. Nếu có cột "Thành tiền" nhưng không có "Đơn giá × Số lượng", hãy điền vào thanh_tien
5. Bỏ qua các dòng tổng cộng/subtotal (chúng là tong_chua_vat, tong_sau_vat)
6. Nếu văn bản bị lỗi OCR nhẹ (ký tự sai), hãy đọc theo ngữ cảnh và tự sửa
7. confidence: 1.0 = chắc chắn, 0.5 = không chắc, 0.3 = đoán

NHẬN DIỆN HEADER BẢNG (tất cả biến thể phổ biến):
- STT, No., #
- Mã hàng, Mã SP, Code, Part No
- Tên hàng, Tên SP, Mô tả, Description, Item
- ĐVT, Đơn vị, Unit, UOM
- Số lượng, SL, Qty, Quantity
- Đơn giá, Giá, Unit Price, Rate
- Thành tiền, Tổng, Amount, Total
- Chiết khấu, CK, Discount
- Ghi chú, Note, Remarks"""


class ExtractionAgent:
    """LangChain-based extraction agent using Gemini Flash with structured output.

    - Retries with exponential backoff on transient API failures.
    - Sends text in paginated chunks to avoid silent data loss on large PDFs.
    - Raises on unrecoverable failures instead of returning empty results.
    """

    def __init__(self, google_api_key: Optional[str] = None):
        self._api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self._api_key,
                temperature=0.1,
            )
        return self._llm

    # ── Public API ─────────────────────────────────────────────

    def extract_from_text(self, ocr_text: str, filename: str = "") -> PriceDocument:
        """
        Extract structured price data from raw OCR text using Gemini Flash.

        For texts longer than MAX_CHARS_PER_CHUNK, pages are processed
        in overlapping chunks and results are merged.
        """
        total_chars = len(ocr_text)
        logger.info(f"Extracting from text ({total_chars} chars)...")

        # Short text — single call
        if total_chars <= MAX_CHARS_PER_CHUNK:
            return self._extract_chunk(ocr_text, filename, chunk_index=0, total_chunks=1)

        # Long text — paginate across chunks
        chunks = self._split_text(ocr_text)
        all_docs: list[PriceDocument] = []
        for i, chunk in enumerate(chunks):
            doc = self._extract_chunk(chunk, filename, chunk_index=i + 1, total_chunks=len(chunks))
            all_docs.append(doc)

        return self._merge_documents(all_docs)

    def extract_from_image(self, image_path: Path) -> PriceDocument:
        """
        Extract directly from image using Gemini Vision (multimodal).
        Used when OCR confidence is very low.
        """
        import base64
        import google.generativeai as genai

        logger.info(f"Gemini Vision extraction from {image_path.name}...")
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        with open(str(image_path), "rb") as f:
            img_bytes = f.read()

        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

Nhìn vào hình ảnh bảng báo giá và trích xuất thông tin. Trả về JSON:
{{
  "nha_cung_cap": "...", "so_bao_gia": "...", "ngay_bao_gia": "...",
  "khach_hang": "...", "don_vi_tien": "VND",
  "items": [{{"stt": 1, "ten_sp": "...", "dvt": "...", "so_luong": 1, "don_gia": 0, "thanh_tien": 0, "confidence": 0.9}}],
  "tong_chua_vat": null, "thue_vat_pct": 10, "tong_sau_vat": null
}}"""

        response = model.generate_content([
            {"mime_type": mime_type, "data": base64.b64encode(img_bytes).decode()},
            prompt
        ])

        return self._parse_response(response.text)

    # ── Core extraction with retry ─────────────────────────────

    def _extract_chunk(self, text_chunk: str, filename: str,
                       chunk_index: int, total_chunks: int) -> PriceDocument:
        """Call the LLM with retry + exponential backoff on transient errors."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chunk_note = f" (trang {chunk_index}/{total_chunks})" if total_chunks > 1 else ""
        user_prompt = f"""Trích xuất thông tin bảng báo giá từ văn bản OCR sau đây.
File: {filename}{chunk_note}

=== NỘI DUNG OCR ===
{text_chunk}
=== HẾT NỘI DUNG ===

Trả về JSON ĐÚNG định dạng sau (không giải thích thêm):
{{
  "nha_cung_cap": "Tên công ty báo giá hoặc null",
  "so_bao_gia": "Số báo giá hoặc null",
  "ngay_bao_gia": "dd/mm/yyyy hoặc null",
  "khach_hang": "Tên khách hàng hoặc null",
  "don_vi_tien": "VND hoặc USD hoặc EUR",
  "items": [
    {{
      "stt": 1,
      "ma_sp": "Mã SP hoặc null",
      "ten_sp": "Tên sản phẩm bắt buộc",
      "dvt": "Đơn vị hoặc null",
      "so_luong": 1.0,
      "don_gia": 100000.0,
      "thanh_tien": 100000.0,
      "chiet_khau_pct": null,
      "ghi_chu": null,
      "confidence": 0.95
    }}
  ],
  "tong_chua_vat": null,
  "thue_vat_pct": 10.0,
  "thue_vat_tien": null,
  "tong_sau_vat": null,
  "dieu_kien_thanh_toan": null,
  "thoi_gian_giao_hang": null,
  "bao_hanh": null,
  "ghi_chu_chung": null
}}"""

        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        llm = self._get_llm()
        last_error: Exception = ValueError("unreachable")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages)
                return self._parse_response(response.content)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                    logger.warning(
                        f"[ExtractionAgent] attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[ExtractionAgent] all {MAX_RETRIES} attempts failed: {e}"
                    )
                    raise   # propagate — don't silently return empty doc

        # Exhausted retries (unreachable but keeps type-checker happy)
        raise last_error

    # ── Text chunking ─────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
        """Split text into overlapping chunks at line boundaries."""
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line)
            if current_len + line_len > max_chars and current:
                chunks.append("\n".join(current))
                # Overlap: carry last 3 lines to avoid splitting mid-table
                overlap = current[-3:]
                current = overlap[:]
                current_len = sum(len(l) for l in current)
            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current))

        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks

    # ── Merge multi-chunk results ─────────────────────────────

    @staticmethod
    def _merge_documents(docs: list[PriceDocument]) -> PriceDocument:
        """Merge metadata from the first doc with items from all chunks."""
        if not docs:
            return PriceDocument(don_vi_tien="VND", items=[])

        base = docs[0]
        merged_items: list[PriceItem] = []
        seen_sps: set[str] = set()

        for doc in docs:
            for item in doc.items:
                # Deduplicate by product name
                key = (item.ten_sp or "").strip().lower()
                if key and key not in seen_sps:
                    seen_sps.add(key)
                    merged_items.append(item)
                elif not key:
                    merged_items.append(item)

        logger.info(f"Merged {len(docs)} chunks → {len(merged_items)} unique items")
        base.items = merged_items
        return base

    # ── Response parsing ───────────────────────────────────────

    def _parse_response(self, content: str) -> PriceDocument:
        """Parse LLM JSON response into PriceDocument."""
        try:
            # Extract JSON from code block if wrapped
            text = content.strip()
            if "```" in text:
                start = text.find("{", text.find("```"))
                end = text.rfind("}") + 1
                text = text[start:end]
            elif text.startswith("{"):
                pass
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

            data = json.loads(text)

            # Parse items
            items: list[PriceItem] = []
            for item_data in data.get("items", []):
                try:
                    item = PriceItem(
                        stt=item_data.get("stt"),
                        ma_sp=item_data.get("ma_sp"),
                        ten_sp=item_data.get("ten_sp", ""),
                        dvt=item_data.get("dvt"),
                        so_luong=_safe_float(item_data.get("so_luong")),
                        don_gia=_safe_float(item_data.get("don_gia")),
                        thanh_tien=_safe_float(item_data.get("thanh_tien")),
                        chiet_khau_pct=_safe_float(item_data.get("chiet_khau_pct")),
                        ghi_chu=item_data.get("ghi_chu"),
                        confidence=float(item_data.get("confidence", 0.8)),
                    )
                    # Auto-calculate thanh_tien if missing
                    if item.so_luong and item.don_gia and not item.thanh_tien:
                        item.thanh_tien = item.so_luong * item.don_gia
                    if item.ten_sp:
                        items.append(item)
                except Exception as e:
                    logger.warning(f"Skipping malformed item: {e}")

            doc = PriceDocument(
                nha_cung_cap=data.get("nha_cung_cap"),
                so_bao_gia=data.get("so_bao_gia"),
                ngay_bao_gia=data.get("ngay_bao_gia"),
                khach_hang=data.get("khach_hang"),
                don_vi_tien=data.get("don_vi_tien", "VND"),
                items=items,
                tong_chua_vat=_safe_float(data.get("tong_chua_vat")),
                thue_vat_pct=_safe_float(data.get("thue_vat_pct")),
                thue_vat_tien=_safe_float(data.get("thue_vat_tien")),
                tong_sau_vat=_safe_float(data.get("tong_sau_vat")),
                dieu_kien_thanh_toan=data.get("dieu_kien_thanh_toan"),
                thoi_gian_giao_hang=data.get("thoi_gian_giao_hang"),
                bao_hanh=data.get("bao_hanh"),
                ghi_chu_chung=data.get("ghi_chu_chung"),
            )
            logger.info(f"Extracted {len(items)} items successfully.")
            return doc

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}\nContent: {content[:500]}")
            # Return partial doc so the pipeline can still save what it has
            return PriceDocument(don_vi_tien="VND", items=[])


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace(".", "").replace(" ", "")
            for suffix in ["VND", "đ", "$", "USD", "EUR"]:
                val = val.replace(suffix, "")
        return float(val)
    except (ValueError, TypeError):
        return None
