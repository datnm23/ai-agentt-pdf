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
Nhiệm vụ: trích xuất TẤT CẢ thông tin từ bảng báo giá — header, items, và footer.

═══════════════════════════════════════════════════════════════════════════════
PHÂN LOẠI TÀI LIỆU (ĐỌC TRƯỚC KHI TRÍCH XUẤT):
═══════════════════════════════════════════════════════════════════════════════
- Nếu thấy "BẢNG GIÁ", "Price List", "Catalog", "Bảng Báo Giá Sản Phẩm" → loai_tai_lieu = "price_list"
  → KHÔNG tính tổng tiền, KHÔNG cộng dồn, so_luong = null (vì đây là bảng tra cứu giá, không phải đơn hàng)
- Nếu thấy "BÁO GIÁ", "Quotation", "Phiếu Báo Giá" → loai_tai_lieu = "quote"
  → Có thể tính tổng nếu có số lượng cụ thể
- Nếu thấy "HÓA ĐƠN", "Invoice", "Phiếu Xuất Kho" → loai_tai_lieu = "invoice"
  → Bắt buộc có tổng tiền

═══════════════════════════════════════════════════════════════════════════════
HEADER (đọc kỹ đầu tài liệu):
═══════════════════════════════════════════════════════════════════════════════
- nha_cung_cap: Tên công ty báo giá (VD: "CÔNG TY CỔ PHẦN ĐÔNG GIANG")
- dia_chi: Địa chỉ công ty (đường, quận, thành phố)
- dien_thoai: Số điện thoại liên hệ (nhiều số nếu có)
- email: Email công ty
- so_bao_gia: Số báo giá / Số chứng từ
- ngay_bao_gia: Ngày lập báo giá (dd/mm/yyyy) — TÌM pattern: "hiệu lực từ ngày", "áp dụng từ", "kể từ ngày"
- ngay_het_han: Ngày hết hạn báo giá
- khach_hang: Tên khách hàng / đơn vị nhận báo giá
- don_vi_tien: Đơn vị tiền tệ (VND, USD, EUR)

═══════════════════════════════════════════════════════════════════════════════
NHÓM SẢN PHẨM (Stateful Parsing - RẤT QUAN TRỌNG):
═══════════════════════════════════════════════════════════════════════════════
- Khi gặp dòng IN ĐẬM / IN HOA đứng độc lập trước bảng → đây là TÊN NHÓM SẢN PHẨM
  VD: "DÂY ĐIỆN 1 LÕI RUỘT MỀM", "CÁP ĐIỀU KHIỂN", "DÂY CÁP ĐỒNG TRẦN"
- Lưu nhóm này vào trường nhom_sp cho TẤT CẢ sản phẩm bên dưới cho đến khi gặp nhóm mới
- VD: Dòng "1 x 1.5" thuộc nhóm "DÂY ĐIỆN 1 LÕI RUỘT MỀM" → nhom_sp = "DÂY ĐIỆN 1 LÕI RUỘT MỀM"
- ĐẶC BIỆT: Nếu ĐẦU TRANG có dòng "SẢN PHẨM: <tên>" hoặc "PRODUCT: <tên>" (thường in đậm/to)
  → đó là nhom_sp cho TẤT CẢ sản phẩm trên trang, dù không có nhóm nào khác
  VD: "SẢN PHẨM: CÁP ĐỒNG TRẦN" → nhom_sp = "CÁP ĐỒNG TRẦN" cho mọi dòng trong bảng

═══════════════════════════════════════════════════════════════════════════════
ITEMS (đọc kỹ từng dòng trong bảng):
═══════════════════════════════════════════════════════════════════════════════
- **nhom_sp: Nhóm sản phẩm** — từ dòng tiêu đề nhóm gần nhất (xem phần trên)
- **ma_sp: Mã sản phẩm / mã hàng** — TUYỆT ĐỐI không bỏ sót! Tìm trong các cột:
  ┌────────────────────────────────────────────────────────────────┐
  │ "Mã SP", "Mã hàng", "Code", "Part No", "SKU", "Ký hiệu",      │
  │ "Product Symbol", "Item Code", "Model", "Part Number"          │
  └────────────────────────────────────────────────────────────────┘
  VD: Cột "Ký hiệu" có giá trị "CV 0.5R5-0.3" → ma_sp = "CV 0.5R5-0.3"
- **ten_sp: Tên sản phẩm** — BẮT BUỘC, giữ nguyên KHÔNG viết tắt
  - VD đúng: "DÂY CÁP ĐIỆN VDeb VCmt 1×2.5mm² ruột đồng 7 sợi"
  - VD sai: "cáp" hoặc "dây" (VIẾT TẮT QUÁ)
- **dvt: Đơn vị tính** — trích xuất từ cột ĐVT
  - Cable/wire: "m" (mét) | Kg: "kg" | Bottle: "chai" | Box: "hộp" | Set: "bộ" | Roll: "cuộn"
- **don_gia: Đơn giá chính** (bỏ dấu chấm phẩy: "1.200.000" → 1200000)
- **dvt_2: Đơn vị tính thứ 2** — nếu bảng có nhiều cột đơn giá (VD: "VND/kg" bên cạnh "VND/m")
- **don_gia_2: Đơn giá thứ 2** — giá theo đơn vị thứ 2
- **vat_pct: % VAT của dòng này** — đọc từ header cột hoặc ghi chú nhóm
  VD: Header "Đơn giá (Đã có 8% VAT)" → vat_pct = 8
- so_luong: Số lượng — nếu loai_tai_lieu = "price_list" → null
- thanh_tien: Thành tiền — nếu loai_tai_lieu = "price_list" → null
- stt: Số thứ tự liên tục 1→N trong toàn bộ tài liệu (KHÔNG reset mỗi trang)
- ghi_chu: Ghi chú (xuất xứ, quy cách, bảo hành...)

═══════════════════════════════════════════════════════════════════════════════
BẢNG CÓ NHIỀU CỘT ĐƠN GIÁ (Dynamic Columns):
═══════════════════════════════════════════════════════════════════════════════
VD header: | Tiết diện | Ký hiệu | Đơn giá VND/m | Đơn giá VND/kg |
→ don_gia = giá VND/m, dvt = "m"
→ don_gia_2 = giá VND/kg, dvt_2 = "kg"

═══════════════════════════════════════════════════════════════════════════════
THUẾ VAT (Xác định chính xác):
═══════════════════════════════════════════════════════════════════════════════
- Đọc header cột đơn giá: "Đã có 8% VAT" → vat_pct = 8 cho TẤT CẢ dòng trong bảng đó
- Đọc header cột đơn giá: "Đã có 10% VAT" → vat_pct = 10 cho TẤT CẢ dòng trong bảng đó
- TUYỆT ĐỐI KHÔNG để vat_pct = null nếu đã thấy % VAT trong header cột
- Nếu ghi "Giá chưa VAT" hoặc không ghi gì → gia_da_bao_gom_vat = false
- Nếu mỗi nhóm/trang có VAT khác nhau → ghi vat_pct đúng cho từng dòng item
- VD: Dây cáp bọc (trang 1–13) = 8%, Cáp đồng trần (trang 14) = 10%

═══════════════════════════════════════════════════════════════════════════════
FOOTER (đọc kỹ cuối tài liệu):
═══════════════════════════════════════════════════════════════════════════════
- tong_chua_vat: Tổng tiền chưa VAT — NẾU loai_tai_lieu = "price_list" → null
- thue_vat_pct: % VAT chung (thường 8 hoặc 10)
- thue_vat_tien: Tiền VAT = tong_chua_vat × thue_vat_pct / 100
- tong_sau_vat: Tổng tiền sau VAT
- dieu_kien_thanh_toan: Điều kiện thanh toán (VD: "CK 5% sau 30 ngày")
- thoi_gian_giao_hang: Thời gian giao hàng
- bao_hanh: Thời hạn bảo hành
- ghi_chu_chung: Ghi chú chung

═══════════════════════════════════════════════════════════════════════════════
NHẬN DIỆN CỘT (tất cả biến thể):
═══════════════════════════════════════════════════════════════════════════════
STT | No.# | Mã hàng.Code.Ký hiệu.Symbol | Tên hàng.Tên SP.Mô tả.Item.Tiết diện |
ĐVT.Đơn vị.Unit | Số lượng.SL.Qty | Đơn giá.Giá.Unit Price.Rate.VND/m.VND/kg |
Thành tiền.Tổng.Amount.Total | Ghi chú.Note

confidence: 1.0=chắc chắn, 0.7-0.9=bình thường, <0.7=cần kiểm tra"""


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
            doc = self._extract_chunk(ocr_text, filename, chunk_index=0, total_chunks=1)
            return self._post_process(doc)

        # Long text — paginate across chunks
        chunks = self._split_text(ocr_text)
        all_docs: list[PriceDocument] = []
        for i, chunk in enumerate(chunks):
            doc = self._extract_chunk(chunk, filename, chunk_index=i + 1, total_chunks=len(chunks))
            all_docs.append(doc)

        merged = self._merge_documents(all_docs)
        return self._post_process(merged)

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

ĐÂY LÀ HÌNH ẢNH. Nhìn kỹ TỪNG GÓC của trang:
- Phần đầu: logo, tên công ty, địa chỉ, điện thoại, email, số báo giá, ngày
- Phần giữa: bảng sản phẩm — đọc TẤT CẢ các dòng, mỗi cột
- Phần cuối: tổng cộng, VAT, điều kiện thanh toán, giao hàng, bảo hành

Trả về JSON đầy đủ:
{{
  "loai_tai_lieu": "price_list | quote | invoice",
  "gia_da_bao_gom_vat": false,
  "nha_cung_cap": "...", "dia_chi": "...", "dien_thoai": "...", "email": "...",
  "so_bao_gia": "...", "ngay_bao_gia": "...", "ngay_het_han": "...",
  "khach_hang": "...", "don_vi_tien": "VND",
  "items": [{{"stt":1,"nhom_sp":"...","ma_sp":"...","ten_sp":"...","dvt":"m","so_luong":null,"don_gia":0,"dvt_2":null,"don_gia_2":null,"thanh_tien":null,"vat_pct":null,"ghi_chu":"...","confidence":0.9}}],
  "tong_chua_vat": null, "thue_vat_pct": 10, "thue_vat_tien": null, "tong_sau_vat": null,
  "dieu_kien_thanh_toan": "...", "thoi_gian_giao_hang": "...", "bao_hanh": "...", "ghi_chu_chung": "..."
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
        user_prompt = f"""⚠️ ĐỌC KỸ TOÀN BỘ văn bản. Tìm thông tin công ty, địa chỉ, điện thoại, email ở ĐẦU trang. Tìm tổng cộng, VAT, điều kiện thanh toán ở CUỐI trang.

Trích xuất thông tin bảng báo giá từ văn bản OCR sau đây.
File: {filename}{chunk_note}

=== NỘI DUNG OCR ===
{text_chunk}
=== HẾT NỘI DUNG ===

Trả về JSON ĐÚNG định dạng sau (không giải thích thêm):
{{
  "loai_tai_lieu": "price_list | quote | invoice (XÁC ĐỊNH TỪ TIÊU ĐỀ TÀI LIỆU)",
  "gia_da_bao_gom_vat": false,
  "nha_cung_cap": "Tên công ty báo giá hoặc null",
  "dia_chi": "Địa chỉ công ty hoặc null",
  "dien_thoai": "Điện thoại hoặc null",
  "email": "Email hoặc null",
  "so_bao_gia": "Số báo giá hoặc null",
  "ngay_bao_gia": "dd/mm/yyyy hoặc null — TÌM 'hiệu lực từ ngày', 'áp dụng từ'",
  "ngay_het_han": "dd/mm/yyyy hoặc null",
  "khach_hang": "Tên khách hàng hoặc null",
  "don_vi_tien": "VND hoặc USD hoặc EUR",
  "items": [
    {{
      "stt": 1,
      "nhom_sp": "TÊN NHÓM SẢN PHẨM từ dòng in đậm/in hoa phía trên (VD: DÂY ĐIỆN 1 LÕI RUỘT MỀM)",
      "ma_sp": "Mã SP / Ký hiệu / Code / Part No — TUYỆT ĐỐI không bỏ sót!",
      "ten_sp": "Tên sản phẩm BẮT BUỘC, giữ nguyên KHÔNG viết tắt",
      "dvt": "Đơn vị tính chính (m, kg, cái...)",
      "so_luong": "null nếu price_list, số lượng nếu quote/invoice",
      "don_gia": 100000.0,
      "dvt_2": "Đơn vị thứ 2 nếu có (VD: kg khi có cột VND/kg)",
      "don_gia_2": "Giá theo đơn vị thứ 2 nếu có",
      "thanh_tien": "null nếu price_list",
      "vat_pct": "% VAT của dòng này nếu khác VAT chung",
      "chiet_khau_pct": null,
      "ghi_chu": null,
      "confidence": 0.95
    }}
  ],
  "tong_chua_vat": "null nếu price_list",
  "thue_vat_pct": 10.0,
  "thue_vat_tien": "null nếu price_list",
  "tong_sau_vat": "null nếu price_list",
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
        """Merge metadata from the first doc with items from all chunks.

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

        logger.info(f"Merged {len(docs)} chunks → {len(merged_items)} unique items")
        base.items = merged_items
        return base

    # ── Response parsing ───────────────────────────────────────

    @staticmethod
    def _inherit_sparse_fields(doc: PriceDocument) -> None:
        """Propagate nhom_sp, dvt, vat_pct from the nearest previous item that has a value.

        Mutates in-place. Call per-page BEFORE merging to prevent cross-page contamination.
        For text-based extraction it is called inside _post_process instead.
        """
        last_group: str | None = None
        last_dvt: str | None = None
        last_vat_pct: float | None = None
        for item in doc.items:
            if item.nhom_sp:
                last_group = item.nhom_sp
            elif last_group:
                item.nhom_sp = last_group
            if item.dvt:
                last_dvt = item.dvt
            elif last_dvt:
                item.dvt = last_dvt
            if item.vat_pct is not None:
                last_vat_pct = item.vat_pct
            elif last_vat_pct is not None:
                item.vat_pct = last_vat_pct

    @staticmethod
    def _post_process(doc: PriceDocument, skip_field_inheritance: bool = False) -> PriceDocument:
        """Fill missing fields with smart defaults — especially dvt, so_luong, thanh_tien, VAT.

        For price_list documents: do NOT calculate totals or set so_luong/thanh_tien.
        skip_field_inheritance: set True when _inherit_sparse_fields was already applied
          per-page (scanned PDF flow) to prevent cross-page contamination.
        """
        is_price_list = doc.loai_tai_lieu == "price_list"

        # 1. Renumber stt globally 1..N
        for i, item in enumerate(doc.items, 1):
            item.stt = i

        # 1b. Propagate sparse fields (skip when already done per-page before merging)
        if not skip_field_inheritance:
            ExtractionAgent._inherit_sparse_fields(doc)

        # 2. For price_list: clear so_luong and thanh_tien (meaningless for catalog)
        if is_price_list:
            for item in doc.items:
                item.so_luong = None
                item.thanh_tien = None
            # Clear document-level totals too
            doc.tong_chua_vat = None
            doc.thue_vat_tien = None
            doc.tong_sau_vat = None
        else:
            # 2b. Auto-calculate thanh_tien if missing (quote/invoice only)
            for item in doc.items:
                if item.so_luong is not None and item.don_gia is not None and item.thanh_tien is None:
                    item.thanh_tien = item.so_luong * item.don_gia

            # 3. Default so_luong=1.0 for per-unit price items (no quantity column) — quote/invoice only
            for item in doc.items:
                if item.so_luong is None and item.don_gia is not None:
                    item.so_luong = 1.0

        # 4. Infer dvt from product name when missing — key for cable/electrical items
        unit_map = [
            # Cable / wire
            ("dây", "m"), ("day", "m"), ("cáp", "m"), ("cap", "m"),
            ("wire", "m"), ("cable", "m"), ("đây điện", "m"),
            ("dây cáp", "m"), ("dây dẫn", "m"),
            # Others
            ("ổ cắm", "cái"), ("công tắc", "cái"), ("bóng", "cái"), ("đèn", "cái"),
            ("vit", "bộ"), ("ống", "m"), ("ruột", "m"),
            ("bình", "cái"), ("máy", "cái"), ("thiết bị", "cái"),
        ]
        for item in doc.items:
            if item.dvt is None and item.ten_sp:
                name_lower = item.ten_sp.lower()
                for kw, inferred_unit in unit_map:
                    if kw in name_lower:
                        item.dvt = inferred_unit
                        break

        # 5. Auto-calculate VAT totals — ONLY for quote/invoice, NOT for price_list
        if not is_price_list:
            has_items = len(doc.items) > 0
            tong_raw = doc.tong_chua_vat
            # Treat 0 as "not extracted" so we can compute from items later
            if has_items and (tong_raw is None or tong_raw == 0):
                # Sum thanh_tien from all items as a fallback total
                doc.tong_chua_vat = sum(
                    (item.thanh_tien or 0) for item in doc.items
                )
            if doc.tong_chua_vat and doc.tong_chua_vat > 0 and doc.thue_vat_pct and doc.thue_vat_pct > 0:
                if doc.thue_vat_tien is None:
                    doc.thue_vat_tien = doc.tong_chua_vat * doc.thue_vat_pct / 100
                if doc.tong_sau_vat is None:
                    doc.tong_sau_vat = doc.tong_chua_vat + (doc.thue_vat_tien or 0)

        return doc

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
                        nhom_sp=item_data.get("nhom_sp"),
                        ma_sp=item_data.get("ma_sp"),
                        ten_sp=item_data.get("ten_sp", ""),
                        dvt=item_data.get("dvt"),
                        so_luong=_safe_float(item_data.get("so_luong")),
                        don_gia=_safe_float(item_data.get("don_gia")),
                        dvt_2=item_data.get("dvt_2"),
                        don_gia_2=_safe_float(item_data.get("don_gia_2")),
                        thanh_tien=_safe_float(item_data.get("thanh_tien")),
                        chiet_khau_pct=_safe_float(item_data.get("chiet_khau_pct")),
                        vat_pct=_safe_float(item_data.get("vat_pct")),
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
                loai_tai_lieu=data.get("loai_tai_lieu"),
                gia_da_bao_gom_vat=data.get("gia_da_bao_gom_vat", False),
                nha_cung_cap=data.get("nha_cung_cap"),
                dia_chi=data.get("dia_chi"),
                dien_thoai=data.get("dien_thoai"),
                email=data.get("email"),
                so_bao_gia=data.get("so_bao_gia"),
                ngay_bao_gia=data.get("ngay_bao_gia"),
                ngay_het_han=data.get("ngay_het_han"),
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
            # NOTE: do NOT call _post_process here — tong_chua_vat is per-page (0 or meaningless)
            # _post_process will be called once on the merged document instead
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
