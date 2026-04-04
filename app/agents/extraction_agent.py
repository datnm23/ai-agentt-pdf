"""AI Extraction Agent — LangChain + Gemini Flash with structured output."""
from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from app.models.schemas import PriceDocument, PriceItem

# ── Constants ────────────────────────────────────────────────────
MAX_CHARS_PER_CHUNK = 12_000   # character-based safety cap (secondary limit)
MAX_ROWS_PER_CHUNK  = 30       # primary limit: max Markdown data rows per chunk
                               # Reduced 50→30: flash-lite output is less token-efficient;
                               # 30 rows → ~3000-4000 output tokens, safe within 65536 limit
MAX_RETRIES = 3
BASE_BACKOFF_SECS = 2.0

# ── DVT normalization map (rule-based, 0 API cost) ────────────────────────────
_DVT_MAP: dict[str, str] = {
    # Cái / chiếc
    "chiếc": "cái", "c": "cái", "ea": "cái", "pcs": "cái",
    "piece": "cái", "cai": "cái",
    # Bộ / set
    "set": "bộ", "bo": "bộ",
    # Mét
    "mét": "m", "met": "m", "metre": "m", "meter": "m",
    # Kg
    "kgs": "kg", "kilo": "kg",
    # Cuộn
    "roll": "cuộn", "cuon": "cuộn",
    # Cây / thanh
    "thanh": "cây", "bar": "cây", "rod": "cây",
    # Tấm / miếng
    "miếng": "tấm", "sheet": "tấm", "tam": "tấm",
    # Lít
    "lit": "lít", "litre": "lít", "liter": "lít",
    # Thùng / hộp
    "hop": "hộp", "thung": "thùng", "box": "hộp",
}


def _normalize_dvt(dvt: str) -> str:
    """Normalize DVT to canonical Vietnamese unit. Returns dvt unchanged if not in map."""
    return _DVT_MAP.get(dvt.strip().lower(), dvt.strip()) if dvt else dvt


EXTRACTION_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích BẢNG GIÁ của các doanh nghiệp Việt Nam.
Nhiệm vụ: trích xuất TẤT CẢ sản phẩm và đơn giá từ bảng giá — header công ty, từng dòng sản phẩm.

═══════════════════════════════════════════════════════════════════════════════
HEADER (đọc kỹ đầu tài liệu):
═══════════════════════════════════════════════════════════════════════════════
- nha_cung_cap: Tên công ty (VD: "CÔNG TY CỔ PHẦN ĐÔNG GIANG")
  Header lặp đầu mỗi trang thường có tên công ty — VD: "VAN VÒI MINH HÒA – VAN VÒI VIỆT..."
  → Trích tên công ty ngắn gọn
- dia_chi: Địa chỉ công ty
- dien_thoai: Số điện thoại
- email: Email công ty
- ngay_hieu_luc: Ngày hiệu lực bảng giá (dd/mm/yyyy) — tìm "hiệu lực từ ngày", "áp dụng từ", "kể từ ngày"
- don_vi_tien: Đơn vị tiền tệ (VND, USD, EUR)
- ghi_chu_chung: Ghi chú chung cuối tài liệu (nếu có)

═══════════════════════════════════════════════════════════════════════════════
NHÓM SẢN PHẨM (Stateful Parsing - RẤT QUAN TRỌNG):
═══════════════════════════════════════════════════════════════════════════════
- Khi gặp dòng IN HOA đứng độc lập trước bảng → đây là TÊN NHÓM SẢN PHẨM
  VD: "DÂY ĐIỆN 1 LÕI RUỘT MỀM", "CÁP ĐIỀU KHIỂN"
- Lưu nhóm này vào nhom_sp cho TẤT CẢ sản phẩm BÊN DƯỚI tiêu đề đó cho đến khi gặp nhóm mới
- ⚠️ QUY TẮC VỊ TRÍ: nhóm chỉ áp dụng cho sản phẩm NẰM DƯỚI tiêu đề nhóm trong ảnh.
  Sản phẩm nằm TRÊN tiêu đề nhóm → thuộc nhóm trước đó (hoặc nhóm từ trang trước nếu có hint).
  KHÔNG được gán ngược nhóm mới cho sản phẩm đã xuất hiện TRƯỚC tiêu đề đó.
- Nếu KHÔNG có tiêu đề nhóm rõ ràng: dùng tên loại sản phẩm chính làm nhom_sp
  VD: bảng chỉ có "VAN CỬA ĐỒNG / MIHA-XK PN20" → nhom_sp = "VAN CỬA ĐỒNG"
- Dòng "### NHÓM SẢN PHẨM: <tên>" trong văn bản → nhom_sp = <tên>
- Dòng "[NHÓM SẢN PHẨM TỪ TRANG TRƯỚC]: <tên>" ở đầu đoạn → dùng làm nhom_sp mặc định
  cho tất cả sản phẩm cho đến khi gặp nhóm mới

═══════════════════════════════════════════════════════════════════════════════
ITEMS (đọc kỹ từng dòng trong bảng):
═══════════════════════════════════════════════════════════════════════════════
╔══════════════════════════════════════════════════════════════════╗
║  2 TRƯỜNG BẮT BUỘC — item KHÔNG có đủ thì BỎ QUA                ║
║  1. ten_sp   — tên sản phẩm (BẮT BUỘC)                          ║
║  2. GIÁ: don_gia  HOẶC  don_gia_co_vat  (BẮT BUỘC)              ║
║                                                                  ║
║  KHÔNG bắt buộc nhưng cần điền nếu có:                          ║
║  - qui_cach: kích thước / thông số kỹ thuật                      ║
║  - dvt: đơn vị tính                                              ║
╚══════════════════════════════════════════════════════════════════╝

- **ma_sp: Mã sản phẩm** — tìm trong cột "Mã SP", "Code", "Ký hiệu", "KÝ HIỆU", "Model", "SKU"
  ĐẶC BIỆT 1: Nếu cột "Tên SP" có 2 dòng (VD: "VAN CỬA ĐỒNG\nMIHA-XK PN20"):
  → ten_sp = "VAN CỬA ĐỒNG", ma_sp = "MIHA-XK PN20" (dòng 2 = model code)
  ĐẶC BIỆT 2: Tên SP KHÔNG có cột MÃ SP riêng nhưng kết thúc bằng mã nhãn hiệu:
  → Pattern: <Tên loại sản phẩm> <BRAND_CODE> PN<số> hoặc DN<số>
  → VD: "RỌ ĐỒNG MBV PN10" → ten_sp="RỌ ĐỒNG", ma_sp="MBV PN10"
  → VD: "VAN BI ĐỒNG TAY GẠT MH PN10" → ten_sp="VAN BI ĐỒNG TAY GẠT", ma_sp="MH PN10"
  → VD: "VAN CỬA ĐỒNG MIHA-XK PN20" → ten_sp="VAN CỬA ĐỒNG", ma_sp="MIHA-XK PN20"
  → Nhãn hiệu thường gặp: MIHA, MBV, MH, MI, DALING, TURA, TUBO + PN/DN + số
- **ten_sp: Tên sản phẩm** ⚠️ BẮT BUỘC — giữ nguyên KHÔNG viết tắt
  - VD đúng: "DÂY ĐIỆN 1 LÕI RUỘT MỀM – 300/500V Cu/PVC"
  - KHÔNG nhét kích thước/quy cách vào ten_sp nếu đã có cột riêng
- **qui_cach: Quy cách / kích thước** — thông số kỹ thuật phân biệt sản phẩm:
  - Tìm trong: "Quy cách", "Tiết diện", "Kích thước", "Spec", "Size", "DN", "PN"
  - Nếu kích thước nằm TRONG tên SP (VD "Dây 1×0.5 Cu/PVC") → tách:
      ten_sp = "Dây Cu/PVC", qui_cach = "1×0.5"
  - Nếu bảng là MA TRẬN (header có DN15, DN20...) → mỗi cột là 1 item riêng:
      qui_cach = tên cột, don_gia = giá tương ứng
  - BẢNG MA TRẬN ĐA NGÀNH — nhiều cột thông số (D/PN, Tiết diện/Điện áp, Công suất/Kích thước):
    BẮT BUỘC ghép [NHÃN CỘT] + [GIÁ TRỊ] vào qui_cach — KHÔNG trích số trần thiếu ngữ cảnh:
      • Cột "D" = "450", cột "PN" = "12.5" → qui_cach = "D450 PN12.5"
      • Cột "Tiết diện" = "3×6", cột "Điện áp" = "1kV" → qui_cach = "3×6mm² 1kV"
    TUYỆT ĐỐI KHÔNG đưa thông số kích thước (D450, 200mm, DN63) vào ma_sp —
    ma_sp chỉ dành cho mã SKU/Model/Code thật sự (VD: "XPS-9320", "MBV-PN10", "MIHA-XK")
  - THÔNG SỐ KỸ THUẬT (kích thước, áp lực, tiết diện, công suất...) LUÔN vào qui_cach,
    không bao giờ để lại trong ten_sp:
    "Ống HDPE PN10 DN63" → ten_sp="Ống HDPE", qui_cach="DN63 PN10"
    "Cáp CXV 3x6+1x4mm²" → ten_sp="Cáp điện CXV", qui_cach="3×6+1×4mm²"
  - ĐẶC BIỆT: cột "Tên SP" CHỈ có tiết diện (VD: "4 x 70", "3 x 95RC")
    → đây là DÒNG SẢN PHẨM HỢP LỆ — ten_sp = giá trị đó, nhom_sp từ hint
- **dvt: Đơn vị tính** — từ cột ĐVT, hoặc suy luận: cáp/dây → "m", bóng/đèn/van → "cái"
- **don_gia: Đơn giá chưa VAT** — bỏ dấu ngàn: "1.200.000" → 1200000
  - Nếu bảng có "Giá chưa VAT" và "Giá có VAT": don_gia = chưa VAT, don_gia_co_vat = có VAT
  - Nếu chỉ có 1 cột giá → extract vào don_gia
- **don_gia_co_vat: Đơn giá đã bao gồm VAT** — extract khi bảng có cột riêng
- **vat_pct: % VAT** — đọc từ header cột (VD: "Đã có 8% VAT" → vat_pct = 8)
  KHÔNG để null nếu đã thấy % VAT trong header
- **dvt_2, don_gia_2** — khi bảng có 2 cột đơn giá (VD: VND/m và VND/kg)
- stt: số thứ tự 1→N toàn tài liệu (KHÔNG reset mỗi trang)
- ghi_chu: xuất xứ, ghi chú đặc biệt (tùy chọn)

═══════════════════════════════════════════════════════════════════════════════
BẢNG CÓ NHIỀU CỘT ĐƠN GIÁ:
═══════════════════════════════════════════════════════════════════════════════
VD: | Tiết diện | Ký hiệu | Đơn giá VND/m | Đơn giá VND/kg |
→ qui_cach = tiết diện, don_gia = VND/m, dvt = "m", don_gia_2 = VND/kg, dvt_2 = "kg"

═══════════════════════════════════════════════════════════════════════════════
THUẾ VAT:
═══════════════════════════════════════════════════════════════════════════════
- Header "Đã có 8% VAT" → vat_pct = 8; "Đã có 10% VAT" → vat_pct = 10
- "Giá chưa VAT" → gia_da_bao_gom_vat = false
- Mỗi nhóm/trang VAT khác nhau → ghi vat_pct đúng cho từng dòng

═══════════════════════════════════════════════════════════════════════════════
NHẬN DIỆN CỘT:
═══════════════════════════════════════════════════════════════════════════════
STT | Mã hàng/Code/Ký hiệu → ma_sp | Tên hàng/Tên SP/Mô tả → ten_sp |
Quy cách/Tiết diện/Kích thước/Spec/DN → qui_cach |
ĐVT/Đơn vị/Unit → dvt |
Đơn giá/Giá/Giá chưa VAT → don_gia | Giá có VAT/Giá sau VAT → don_gia_co_vat |
Ghi chú/Note → ghi_chu

═══════════════════════════════════════════════════════════════════════════════
BẢNG NHIỀU CỘT SONG SONG (MULTI-COLUMN LAYOUT):
═══════════════════════════════════════════════════════════════════════════════
Một số bảng giá in 2-4 nhóm cột song song để tiết kiệm giấy. VD:
| Mã | Tên SP | PS | Giá | || Mã | Tên SP | PS | Giá | || Mã | Tên SP | PS | Giá |
→ BẮT BUỘC đọc TẤT CẢ nhóm cột từ TRÁI SANG PHẢI — đừng bỏ cột bên phải
→ Mỗi dòng trong mỗi nhóm cột = 1 sản phẩm riêng biệt
→ Cột "PS" hoặc "Áp suất" thường chứa thông số áp suất → qui_cach
→ Đếm số lần lặp tiêu đề cột, extract đủ số lần đó

confidence: 1.0=chắc chắn, 0.7-0.9=bình thường, <0.7=cần kiểm tra"""

VALIDATION_SYSTEM_PROMPT = """Bạn là CHUYÊN GIA CHUẨN HÓA DỮ LIỆU ĐA NGÀNH (Data Normalizer).
Dữ liệu nhận được từ OCR có thể bị sai vị trí hoặc sai định dạng giữa các trường.

═══════════════════════════════════════════════════════════════════════
QUY TẮC 1 — GIẢI CỨU THÔNG SỐ ĐI LẠC (QUAN TRỌNG NHẤT):
═══════════════════════════════════════════════════════════════════════
Quét trường "ma" (Mã SP) và "ten" (Tên SP). Nếu chứa thông số kỹ thuật
(VD: D450, PN16, 16mm², 9000BTU, 1.5kW, DN63) thay vì mã SKU thật:
  → Cắt ra, chuyển nối vào "qc" (qui_cach)
  → Trả về trường bị cắt với giá trị "" (rỗng)

Phân biệt thông số vs mã SKU:
  ✗ Thông số (phải cắt sang qc): "D450", "PN12.5", "16mm²", "9000BTU", "DN63", "D200"
  ✓ Mã SKU thật (giữ nguyên): "XPS-9320", "MBV-PN10", "MIHA-XK", "CXV-3x185"

Ví dụ:
  ma="D450", qc="PN12.5"  → ma_sp="",  qui_cach="D450 PN12.5"
  ma="D560", qc="8"       → ma_sp="",  qui_cach="D560 PN8"
  ten="D200", qc="6"      → ten_sp="Cút đầu hàn" [từ nhom], qui_cach="D200 PN6"

═══════════════════════════════════════════════════════════════════════
QUY TẮC 2 — GỌT DŨA ĐỊNH DẠNG QC:
═══════════════════════════════════════════════════════════════════════
Biến chuỗi "Key: Value, Key: Value" thành định dạng kỹ thuật ngắn gọn:
  "D (Ø ngoài): 450, PN: 12.5"       → "D450 PN12.5"
  "Tiết diện: 3x6+1x4, Điện áp: 1kV" → "3×6+1×4mm² 1kV"
  "Công suất: 9000, Đơn vị: BTU"     → "9000BTU"

═══════════════════════════════════════════════════════════════════════
QUY TẮC 3 — BÓC TÁCH THÔNG SỐ DÍNH TRONG TEN_SP:
═══════════════════════════════════════════════════════════════════════
Nếu "ten" còn chứa thông số kỹ thuật lẫn tên gọi:
  ten="Ống HDPE PN10 DN63"       → ten_sp="Ống HDPE",        qui_cach="DN63 PN10"
  ten="Cáp CXV 3x6+1x4mm2"      → ten_sp="Cáp điện CXV",   qui_cach="3×6+1×4mm²"
  ten="Côn thu DN110/DN90 PN10"  → ten_sp="Côn thu",         qui_cach="DN110×DN90 PN10"
  ten="Tôn lạnh 0.45×1200mm"    → ten_sp="Tôn lạnh",        qui_cach="0.45×1200mm"

═══════════════════════════════════════════════════════════════════════
QUY TẮC 4 — BỔ SUNG DVT CÒN THIẾU (BẮT BUỘC):
═══════════════════════════════════════════════════════════════════════
"dvt" là bắt buộc. Nếu trường "dvt" rỗng (""), suy từ ten/nhom/qc:
  Ống, Cáp, Dây, Máng cáp, Thang cáp, Ty ren  → "m"
  Cuộn băng keo, Cuộn dây                       → "cuộn"
  Cút, Tê, Co, Van, Khớp, Côn, Bích, Đầu nối,
    Aptomat, CB, MCB, MCCB, Bóng đèn, Bơm      → "cái"
  Đèn LED (bộ đèn hoàn chỉnh), Bộ điều hòa     → "bộ"
  Tấm, Miếng, Sheet                             → "tấm"
  Không xác định được                            → "cái"

═══════════════════════════════════════════════════════════════════════
QUY TẮC BẢO TỒN:
═══════════════════════════════════════════════════════════════════════
• CHỈ sửa khi chắc chắn >90%. Không chắc → GIỮ NGUYÊN
• KHÔNG tự bịa thông số không có trong dữ liệu gốc
• KHÔNG sửa qc đã chuẩn rồi
• qui_cach="PHỤ KIỆN HDPE" (tên nhóm) → xóa về ""
• qui_cach="Đơn giá (VNĐ)" (tiêu đề cột) → xóa về ""

Trả về JSON array — CHỈ item CẦN SỬA (copy đúng "id", chỉ trả về trường có thay đổi):
[{"id":"r0042","ten_sp":"Cút đầu hàn","qui_cach":"D450 PN12.5","ma_sp":"","dvt":"cái"}]
Không lỗi → []"""


class ExtractionAgent:
    """LangChain-based extraction agent using Gemini Flash with structured output.

    - Retries with exponential backoff on transient API failures.
    - Sends text in paginated chunks to avoid silent data loss on large PDFs.
    - Raises on unrecoverable failures instead of returning empty results.
    """

    # Class-level semaphore — shared across ALL ExtractionAgent instances and all concurrent jobs.
    # Caps total concurrent validation API calls at 5 regardless of how many jobs are running.
    _val_sem: asyncio.Semaphore = asyncio.Semaphore(5)

    def __init__(self, google_api_key: Optional[str] = None):
        _default_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self._text_api_key = os.getenv("GOOGLE_API_KEY_TEXT") or _default_key
        self._vision_api_key = os.getenv("GOOGLE_API_KEY_VISION") or _default_key
        self._llm = None
        self._vision_llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite")
            self._llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self._text_api_key,
                temperature=0.1,
                max_output_tokens=65536,  # max available — prevent output truncation
                thinking_budget=0,        # disable thinking to preserve output token budget
            )
        return self._llm

    def _get_vision_llm(self):
        if self._vision_llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
            self._vision_llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self._vision_api_key,
                temperature=0.1,
                max_output_tokens=65536,  # max available — prevent output truncation
                thinking_budget=0,
            )
        return self._vision_llm

    # ── Public API ─────────────────────────────────────────────

    def extract_from_text(self, ocr_text: str, filename: str = "") -> PriceDocument:
        """
        Extract structured price data from raw OCR text using Gemini Flash.

        For texts longer than MAX_CHARS_PER_CHUNK, pages are processed
        in overlapping chunks and results are merged.
        """
        total_chars = len(ocr_text)
        logger.info(f"Extracting from text ({total_chars} chars)...")

        # Short text — single call (check both char count and row count)
        total_rows = sum(1 for l in ocr_text.split("\n") if ExtractionAgent._is_data_row(l))
        if total_chars <= MAX_CHARS_PER_CHUNK and total_rows <= MAX_ROWS_PER_CHUNK:
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

    def extract_from_image(self, image_path: Path,
                           hint_nhom_sp: Optional[str] = None) -> PriceDocument:
        """
        Extract directly from image using Gemini Vision (multimodal).

        hint_nhom_sp: the last nhom_sp seen on the previous page. When provided,
        it is injected into the prompt so Gemini can correctly assign nhom_sp to
        items at the top of continuation pages where no group header is visible.
        """
        import base64
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        logger.info(f"Gemini Vision extraction from {image_path.name}..."
                    + (f" [hint: {hint_nhom_sp}]" if hint_nhom_sp else ""))

        with open(str(image_path), "rb") as f:
            img_bytes = f.read()

        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        img_b64 = base64.b64encode(img_bytes).decode()

        hint_block = ""
        if hint_nhom_sp:
            hint_block = (
                f"\n⚠️ NHÓM SẢN PHẨM TRANG TRƯỚC: \"{hint_nhom_sp}\"\n"
                "→ QUY TẮC PHÂN NHÓM THEO VỊ TRÍ:\n"
                "   TRƯỜNG HỢP A — Trang bắt đầu TRỰC TIẾP bằng tiêu đề nhóm mới (tiêu đề ở DÒNG ĐẦU TIÊN):\n"
                "   → TOÀN BỘ sản phẩm bên dưới tiêu đề đó thuộc nhóm mới, KHÔNG dùng hint trên.\n"
                "   TRƯỜNG HỢP B — Trang bắt đầu bằng DÒNG SẢN PHẨM (không có tiêu đề ở đầu trang):\n"
                f"   → Các sản phẩm ở ĐẦU TRANG (trước tiêu đề nhóm tiếp theo nếu có) "
                f"thuộc nhóm \"{hint_nhom_sp}\".\n"
                "   → Khi gặp tiêu đề nhóm mới ở GIỮA/CUỐI trang: chỉ sản phẩm BÊN DƯỚI tiêu đề đó "
                "mới thuộc nhóm mới — sản phẩm TRÊN tiêu đề vẫn giữ nhóm hint.\n"
                "   ⛔ KHÔNG gán nhóm mới cho sản phẩm nằm TRÊN tiêu đề nhóm mới.\n"
                "   ⛔ TÊN CÔNG TY / THƯƠNG HIỆU / LOGO / WATERMARK / CON DẤU ở cuối/góc trang "
                "TUYỆT ĐỐI KHÔNG được dùng làm nhom_sp — đây là branding, không phải tiêu đề nhóm.\n"
                f"   ⛔ NẾU KHÔNG có tiêu đề nhóm mới nào trên trang này → dùng nguyên \"{hint_nhom_sp}\" cho TẤT CẢ sản phẩm.\n"
            )

        prompt_text = f"""{EXTRACTION_SYSTEM_PROMPT}
{hint_block}
ĐÂY LÀ HÌNH ẢNH. Nhiệm vụ QUAN TRỌNG NHẤT: trích xuất TẤT CẢ các dòng trong bảng giá.

⚠️ QUY TẮC BẮT BUỘC:
1. STAMP / CON DẤU / CHỮ KÝ / FOOTER: BỎ QUA hoàn toàn — chúng KHÔNG phải sản phẩm
2. TRANG TIẾP THEO (STT không bắt đầu từ 1): đây là trang tiếp nối, vẫn phải extract TẤT CẢ dòng
3. TÊN SẢN PHẨM chỉ là tiết diện (VD: "4 x 70", "3 x 95", "Ngầm 9 x 2.5"):
   → ĐÂY LÀ DÒNG SẢN PHẨM HỢP LỆ — PHẢI extract, ten_sp = giá trị đó, nhom_sp = hint ở trên
4. Nếu stamp/con dấu che khuất một ô giá → đọc giá trị ở vùng xung quanh, confidence = 0.6
5. KHÔNG được trả về items rỗng nếu bảng còn dòng dữ liệu

Nhìn kỹ TỪNG GÓC của trang:
- Phần đầu: logo, tên công ty, địa chỉ, điện thoại, email, ngày hiệu lực bảng giá
- Phần giữa: bảng sản phẩm — đọc TẤT CẢ các dòng kể cả dòng cuối bảng

Trả về JSON:
{{
  "gia_da_bao_gom_vat": false,
  "nha_cung_cap": "...", "dia_chi": "...", "dien_thoai": "...", "email": "...",
  "ngay_hieu_luc": "...", "don_vi_tien": "VND",
  "items": [{{"stt":1,"nhom_sp":"...","ma_sp":"...","ten_sp":"...","qui_cach":"1×0.5","dvt":"m","don_gia":497228,"don_gia_co_vat":537007,"dvt_2":null,"don_gia_2":null,"vat_pct":8,"ghi_chu":null,"confidence":0.9}}],
  "ghi_chu_chung": "..."
}}"""

        llm = self._get_vision_llm()
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
            {"type": "text", "text": prompt_text},
        ])
        response = llm.invoke([message])
        _log_token_usage(response, f"vision:{image_path.name}")
        return self._parse_response(response.content)

    def extract_from_image_halves(self, image_path: Path,
                                   hint_nhom_sp: Optional[str] = None) -> PriceDocument:
        """Split image into left/right halves and extract from each half.

        Fallback for dense multi-column layouts (2-4 parallel sub-tables per page)
        where Gemini Vision misses the rightmost columns when given the full page.
        Each half is sent as a separate Vision call, then results are merged.
        """
        from PIL import Image as PILImage

        with PILImage.open(str(image_path)) as img:
            width, height = img.size
            mid = width // 2
            left_img = img.crop((0, 0, mid, height))
            right_img = img.crop((mid, 0, width, height))

        suffix = image_path.suffix.lower()
        left_path = image_path.parent / f"_halfl_{image_path.stem}{suffix}"
        right_path = image_path.parent / f"_halfr_{image_path.stem}{suffix}"

        try:
            # Save as JPEG to keep file size manageable
            save_fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
            left_img.save(str(left_path), format=save_fmt, quality=90)
            right_img.save(str(right_path), format=save_fmt, quality=90)

            logger.info(f"  → left half: {left_path.name}")
            doc_left = self.extract_from_image(left_path, hint_nhom_sp)
            logger.info(f"  → left: {len(doc_left.items)} items")

            # Carry nhom_sp from left half as hint to right half
            right_hint = next(
                (item.nhom_sp for item in reversed(doc_left.items) if item.nhom_sp),
                hint_nhom_sp,
            )
            logger.info(f"  → right half: {right_path.name} [hint: {right_hint}]")
            doc_right = self.extract_from_image(right_path, right_hint)
            logger.info(f"  → right: {len(doc_right.items)} items")

            if not doc_left.items and not doc_right.items:
                # Both halves empty — try quadrant tiling as last resort
                logger.warning("Both halves returned 0 items — falling back to quadrant tiling")
                return self._extract_from_image_quadrants(image_path, hint_nhom_sp)

            docs = [d for d in (doc_left, doc_right) if d.items]
            return self._merge_documents(docs)
        finally:
            for p in (left_path, right_path):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    def _extract_from_image_quadrants(self, image_path: Path,
                                      hint_nhom_sp: Optional[str] = None) -> PriceDocument:
        """Split image into 4 quadrants (TL, TR, BL, BR) and extract each.

        Last-resort fallback when full-page and half-page Vision calls both fail.
        Handles extremely dense pages where even a half-page has too many items
        for the Vision model to capture completely.
        """
        from PIL import Image as PILImage

        logger.info(f"Quadrant tiling for {image_path.name}")

        with PILImage.open(str(image_path)) as img:
            w, h = img.size
            mx, my = w // 2, h // 2
            regions = {
                "tl": (0,  0,  mx, my),
                "tr": (mx, 0,  w,  my),
                "bl": (0,  my, mx, h),
                "br": (mx, my, w,  h),
            }
            crops = {name: img.crop(box) for name, box in regions.items()}

        suffix = image_path.suffix.lower()
        save_fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
        tmp_paths: list[Path] = []
        docs: list[PriceDocument] = []
        current_hint = hint_nhom_sp

        try:
            for name, crop_img in crops.items():
                tmp = image_path.parent / f"_quad_{name}_{image_path.stem}{suffix}"
                crop_img.save(str(tmp), format=save_fmt, quality=90)
                tmp_paths.append(tmp)

                doc = self.extract_from_image(tmp, current_hint)
                logger.info(f"  quadrant {name}: {len(doc.items)} items")
                if doc.items:
                    docs.append(doc)
                    current_hint = next(
                        (item.nhom_sp for item in reversed(doc.items) if item.nhom_sp),
                        current_hint,
                    )
        finally:
            for p in tmp_paths:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        return self._merge_documents(docs) if docs else PriceDocument(don_vi_tien="VND", items=[])

    # ── Core extraction with retry ─────────────────────────────

    def _extract_chunk(self, text_chunk: str, filename: str,
                       chunk_index: int, total_chunks: int,
                       _depth: int = 0) -> PriceDocument:
        """Call the LLM with retry + exponential backoff on transient errors.

        When output is truncated (finish_reason=MAX_TOKENS), the chunk is bisected
        and each half is extracted independently (max recursion depth=2).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        chunk_note = f" (phần {chunk_index}/{total_chunks})" if total_chunks > 1 else ""
        if _depth:
            chunk_note += f" [bisect depth={_depth}]"

        user_prompt = f"""Trích xuất TẤT CẢ sản phẩm từ bảng giá sau.
File: {filename}{chunk_note}

=== NỘI DUNG ===
{text_chunk}
=== HẾT NỘI DUNG ===

Trả về JSON ĐÚNG định dạng (không giải thích thêm):
{{
  "gia_da_bao_gom_vat": false,
  "nha_cung_cap": "Tên công ty hoặc null",
  "dia_chi": null,
  "dien_thoai": null,
  "email": null,
  "ngay_hieu_luc": "dd/mm/yyyy hoặc null",
  "don_vi_tien": "VND",
  "items": [
    {{
      "stt": 1,
      "nhom_sp": "TÊN NHÓM từ dòng in hoa phía trên (VD: DÂY ĐIỆN 1 LÕI RUỘT MỀM)",
      "ma_sp": "Mã SP / Ký hiệu / Code nếu có",
      "ten_sp": "Tên sản phẩm BẮT BUỘC",
      "qui_cach": "1×0.5 hoặc DN15 hoặc null",
      "dvt": "m hoặc cái hoặc kg...",
      "don_gia": 100000.0,
      "don_gia_co_vat": null,
      "dvt_2": null,
      "don_gia_2": null,
      "vat_pct": 8,
      "ghi_chu": null,
      "confidence": 0.95
    }}
  ],
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
                _log_token_usage(response, chunk_note or "text")

                # Detect output truncation — bisect and recurse rather than lose items
                if _is_truncated(response) and _depth < 2:
                    logger.warning(
                        f"[ExtractionAgent] output truncated at depth={_depth}, bisecting chunk..."
                    )
                    return self._bisect_and_extract(
                        text_chunk, filename, chunk_index, total_chunks, _depth
                    )

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

    def _bisect_and_extract(self, text_chunk: str, filename: str,
                            chunk_index: int, total_chunks: int,
                            depth: int) -> PriceDocument:
        """Split text_chunk in half by line count and extract each half."""
        lines = text_chunk.split("\n")
        mid = len(lines) // 2
        # Snap mid to a non-data-row boundary so we don't split a table row
        while mid < len(lines) and ExtractionAgent._is_data_row(lines[mid]):
            mid += 1

        part1 = "\n".join(lines[:mid])
        part2 = "\n".join(lines[mid:])

        logger.info(f"  bisect → part1: {len(part1)} chars, part2: {len(part2)} chars")
        doc1 = self._extract_chunk(part1, filename, chunk_index * 2 - 1,
                                   total_chunks * 2, depth + 1)
        doc2 = self._extract_chunk(part2, filename, chunk_index * 2,
                                   total_chunks * 2, depth + 1)
        return self._merge_documents([doc1, doc2])

    # ── Pass 2: Validation / qui_cach correction ──────────────

    async def validate_and_correct(self, doc: PriceDocument) -> PriceDocument:
        """Pass 2 (async): DVT normalisation + qui_cach/ten_sp correction via flash-lite.

        Steps:
          1. Build condensed batches (id, nhom, ma, ten, qc) — 30 items each.
          2. Send batches to flash-lite Data Normalizer with UUID keys.
          3. Apply corrections (ten_sp, qui_cach, ma_sp) with idempotent audit trail in ghi_chu.

        Uses class-level _val_sem (Semaphore 5) — shared globally across all concurrent jobs.
        """
        import re as _re2
        if not doc.items:
            return doc

        # 1. Build condensed list with UUID keys
        # "ma" (ma_sp) is included so Pass 2 can spot technical specs stranded in the code field
        id_map: dict[str, int] = {}
        condensed_all: list[dict] = []
        for i, item in enumerate(doc.items):
            val_id = f"r{i:04d}"
            id_map[val_id] = i
            condensed_all.append({
                "id":   val_id,
                "nhom": item.nhom_sp or "",
                "ma":   item.ma_sp or "",    # Pass 2 checks for stranded specs (e.g. "D450")
                "ten":  item.ten_sp or "",
                "qc":   item.qui_cach or "",
                "dvt":  item.dvt or "",      # Pass 2 can correct wrong/missing DVT
            })

        # 4. Batch 30 items; rate-limited by class-level semaphore
        batch_size = 30
        batches = [condensed_all[s:s + batch_size]
                   for s in range(0, len(condensed_all), batch_size)]
        total = len(doc.items)

        async def _run_with_sem(b: list[dict]) -> dict[str, dict]:
            async with self.__class__._val_sem:
                return await self._validate_batch_async(b, total)

        results = await asyncio.gather(
            *[_run_with_sem(b) for b in batches],
            return_exceptions=True,
        )

        # 5. Collect corrections keyed by UUID
        corrections: dict[str, dict] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"[Validation] batch failed: {result}")
                continue
            corrections.update(result)

        if not corrections:
            logger.info("[Validation] No corrections needed.")
            return doc

        logger.info(f"[Validation] Applying {len(corrections)} corrections")

        # 6. Apply corrections + idempotent audit trail
        corrected = list(doc.items)
        for val_id, fix in corrections.items():
            idx = id_map.get(val_id)
            if idx is None:
                continue
            item = corrected[idx]
            changes = []
            # v is not None: allow LLM to clear a noisy field with ""
            if "ten_sp" in fix and fix["ten_sp"] is not None and fix["ten_sp"] != item.ten_sp:
                changes.append(f"ten: '{item.ten_sp}'→'{fix['ten_sp']}'")
            if "qui_cach" in fix and fix["qui_cach"] is not None and fix["qui_cach"] != item.qui_cach:
                changes.append(f"qc: '{item.qui_cach or ''}'→'{fix['qui_cach']}'")
            if "ma_sp" in fix and fix["ma_sp"] is not None and fix["ma_sp"] != (item.ma_sp or ""):
                changes.append(f"ma: '{item.ma_sp or ''}'→'{fix['ma_sp']}'")
            if "dvt" in fix and fix["dvt"] is not None and fix["dvt"] != (item.dvt or ""):
                changes.append(f"dvt: '{item.dvt or ''}'→'{fix['dvt']}'")
            if changes:
                audit = "[Pass2: " + ", ".join(changes) + "]"
                # Idempotency: strip any old Pass2 tag before writing the fresh one
                clean_ghi_chu = _re2.sub(r'\[Pass2:[^\]]+\]\s*', '', item.ghi_chu or "").strip()
                new_ghi_chu = f"{audit} {clean_ghi_chu}".strip()
                update = {k: v for k, v in fix.items()
                          if k in ("ten_sp", "qui_cach", "ma_sp", "dvt") and v is not None}
                update["ghi_chu"] = new_ghi_chu
                corrected[idx] = item.model_copy(update=update)

        result = doc.model_copy(update={"items": corrected})
        if len(result.items) != len(doc.items):
            logger.error(
                f"[Validation] ITEM COUNT MISMATCH: input={len(doc.items)} output={len(result.items)}"
            )
        return result

    async def _validate_batch_async(self, condensed: list[dict],
                                    total: int) -> dict[str, dict]:
        """Send one batch to flash-lite; return {uuid: {ten_sp?, qui_cach?}} corrections.

        Retries up to 3 times with exponential backoff (2s, 4s).
        LLM refusal or un-parseable JSON raises ValueError → triggers retry.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        ids = [c["id"] for c in condensed]
        label = f"validation [{ids[0]}–{ids[-1]}]/{total}"
        prompt = (
            f"Chuẩn hóa dữ liệu cho {len(condensed)} sản phẩm — phát hiện và sửa thông số đi lạc:\n"
            "• 'ma': Nếu chứa thông số kỹ thuật (D450, PN16, DN63...) thay vì mã SKU → cắt sang 'qc', trả về ma_sp=''\n"
            "• 'ten': Nếu chứa thông số lẫn tên gọi → bóc tách sang 'qc'\n"
            "• 'qc': Nếu dạng 'Key: Value' → chuẩn hóa thành 'D450 PN12.5'\n"
            "• 'dvt': BẮT BUỘC — nếu rỗng ('') hãy suy từ ten/nhom/qc và điền vào\n\n"
            + json.dumps(condensed, ensure_ascii=False, indent=2)
            + '\n\nTrả về JSON array (CHỈ item CẦN SỬA — copy đúng "id"):\n'
              '[{"id":"r0042","ten_sp":"Cút đầu hàn","qui_cach":"D450 PN12.5","ma_sp":"","dvt":"cái"}]\n'
              'Không có lỗi → []'
        )
        messages = [
            SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        for attempt in range(3):
            try:
                response = await self._get_llm().ainvoke(messages)
                _log_token_usage(response, label)

                text = response.content.strip()
                a, z = text.find("["), text.rfind("]") + 1
                if a < 0 or z <= a:
                    # LLM refused / hallucinated — no JSON brackets → trigger retry
                    raise ValueError("LLM response missing JSON array brackets")

                raw = text[a:z]
                try:
                    corrections_list = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        recovered = _recover_partial_json(raw)
                        if not isinstance(recovered, list):
                            raise ValueError(f"Recovered JSON is not a list: {type(recovered)}")
                        corrections_list = recovered
                    except Exception as parse_err:
                        raise ValueError(f"JSON parse failed: {parse_err}")  # → triggers retry

                # Accept ten_sp / qui_cach / ma_sp / dvt; v is not None allows clearing with ""
                return {
                    item["id"]: {k: v for k, v in item.items()
                                 if k in ("ten_sp", "qui_cach", "ma_sp", "dvt") and v is not None}
                    for item in corrections_list
                    if isinstance(item.get("id"), str) and item["id"].startswith("r")
                }

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt  # 2s then 4s
                    logger.warning(
                        f"[Validation] {label} attempt {attempt + 1} failed: {e}. "
                        f"Retry in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.warning(
                        f"[Validation] {label} failed after 3 attempts — skipping batch"
                    )
                    return {}  # items left as-is; never crash the whole job
        return {}

    # ── Text chunking ─────────────────────────────────────────

    @staticmethod
    def _detect_table_header(text: str) -> str:
        """Find the table column header line (contains STT/Mã/Tên/Đơn giá...) in text.

        Searches only the first 30 lines to avoid false positives from product rows.
        Returns the header line if found, empty string otherwise.
        """
        HEADER_KEYWORDS = {"stt", "mã", "tên", "đvt", "đơn giá", "số lượng",
                           "thành tiền", "unit", "price", "qty", "description"}
        for line in text.split("\n")[:30]:
            hits = sum(1 for kw in HEADER_KEYWORDS if kw in line.lower())
            if hits >= 3:
                return line
        return ""

    @staticmethod
    def _is_data_row(line: str) -> bool:
        """True when `line` is a Markdown table data row (not header/separator)."""
        stripped = line.strip()
        if not stripped.startswith("|"):
            return False
        # Separator rows like |---|---| or header rows containing column names
        if "---" in stripped:
            return False
        upper = stripped.upper()
        if any(kw in upper for kw in ("TÊN - HÌNH ẢNH", "TÊN SẢN PHẨM", "MÃ SP",
                                       "ĐƠN GIÁ", "STT", "DVT", "ĐVT")):
            return False
        return True

    @staticmethod
    def _split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK,
                    max_rows: int = MAX_ROWS_PER_CHUNK) -> list[str]:
        """Split text into overlapping chunks bounded by row count AND char count.

        Primary limit: MAX_ROWS_PER_CHUNK Markdown data rows per chunk.
        Secondary limit: MAX_CHARS_PER_CHUNK chars (safety cap for non-table text).

        When chunking occurs, injects the table column header and last-seen
        section heading into chunks 2..N so the LLM always has full context.
        """
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        current_rows = 0

        for line in lines:
            line_len = len(line) + 1
            is_data = ExtractionAgent._is_data_row(line)
            # Split when EITHER limit is reached (and we have content already)
            if current and (
                (is_data and current_rows >= max_rows)
                or (current_len + line_len > max_chars)
            ):
                chunks.append("\n".join(current))
                # Overlap: carry last 3 lines to preserve table context
                overlap = current[-3:]
                current = overlap[:]
                current_len = sum(len(l) + 1 for l in current)
                current_rows = sum(1 for l in current if ExtractionAgent._is_data_row(l))
            current.append(line)
            current_len += line_len
            if is_data:
                current_rows += 1

        if current:
            chunks.append("\n".join(current))

        # Inject table-column header AND last-seen section heading into chunks 2..N
        # so the LLM always knows (a) column names and (b) which product group it's in.
        if len(chunks) > 1:
            header_line = ExtractionAgent._detect_table_header(chunks[0])
            current_section: str = ""
            for i in range(len(chunks) - 1):
                # Track the last "NHÓM SẢN PHẨM" declaration seen in chunk i
                for ln in chunks[i].split("\n"):
                    if ln.startswith("### NHÓM SẢN PHẨM:"):
                        current_section = ln.replace("### NHÓM SẢN PHẨM:", "").strip()
                # Build prefix for chunk i+1
                prefix_parts: list[str] = []
                if current_section:
                    prefix_parts.append(
                        f"[NHÓM SẢN PHẨM TỪ TRANG TRƯỚC]: {current_section}"
                    )
                if header_line:
                    prefix_parts.append(
                        f"[HEADER BẢNG TỪ TRANG TRƯỚC]: {header_line}"
                    )
                if prefix_parts:
                    chunks[i + 1] = "\n".join(prefix_parts) + "\n" + chunks[i + 1]

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
        # Key: (ten_sp, qui_cach, don_gia) — preserves same-name items with different specs
        seen_keys: set[tuple] = set()

        for doc in docs:
            for item in doc.items:
                name_key = (item.ten_sp or "").strip().lower()
                spec_key = (item.qui_cach or "").strip().lower()
                # Use don_gia_co_vat as fallback when don_gia is absent (e.g. tables with
                # only a VAT-inclusive price column) — prevents same-item deduplication.
                price_key = round(item.don_gia if item.don_gia is not None
                                  else (item.don_gia_co_vat or 0.0), 2)
                key = (name_key, spec_key, price_key)
                if name_key and key not in seen_keys:
                    seen_keys.add(key)
                    merged_items.append(item)
                elif not name_key:
                    merged_items.append(item)

        # Re-group by nhom_sp in first-occurrence order so that items from the
        # same group (split across chunk boundaries) appear consecutively.
        nhom_order: dict[str, int] = {}
        for i, item in enumerate(merged_items):
            nhom = item.nhom_sp or ""
            if nhom not in nhom_order:
                nhom_order[nhom] = i
        merged_items.sort(key=lambda item: nhom_order.get(item.nhom_sp or "", len(merged_items)))

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
    def _review_null_fields(doc: PriceDocument) -> None:
        """Smart null-field filler: fill nulls caused by page-breaks, leave genuinely absent.

        Rule: within a nhom_sp group (or document-wide if no groups), if ≥1 item has a
        value for a field, forward-fill / back-fill nulls from the nearest non-null neighbour.
        If ALL items in the scope have null, the table genuinely lacks that column — skip.

        Fields reviewed: dvt, vat_pct, qui_cach, nhom_sp (already done), ma_sp (skip — unique).
        """
        items = doc.items
        if not items:
            return

        # ── Helper: fill a field within a contiguous window ──────────────────
        def _fill_window(window: list, get_fn, set_fn):
            """Forward-fill then backward-fill within a window if any value exists."""
            values = [get_fn(it) for it in window]
            if not any(v for v in values):
                return  # all null in this scope → genuinely absent
            # Forward-fill
            last = None
            for it, v in zip(window, values):
                if v:
                    last = v
                elif last:
                    set_fn(it, last)
            # Backward-fill (catches nulls at the start)
            last = None
            for it in reversed(window):
                v = get_fn(it)
                if v:
                    last = v
                elif last:
                    set_fn(it, last)

        # ── Group items by nhom_sp ────────────────────────────────────────────
        from itertools import groupby
        groups: list[list] = []
        for _, grp in groupby(items, key=lambda it: it.nhom_sp or "__none__"):
            groups.append(list(grp))

        # ── dvt ──────────────────────────────────────────────────────────────
        for grp in groups:
            _fill_window(grp,
                         get_fn=lambda it: it.dvt,
                         set_fn=lambda it, v: setattr(it, 'dvt', v))

        # ── vat_pct: fill within group ────────────────────────────────────────
        for grp in groups:
            _fill_window(grp,
                         get_fn=lambda it: it.vat_pct,
                         set_fn=lambda it, v: setattr(it, 'vat_pct', v))

        # ── qui_cach ─────────────────────────────────────────────────────────
        # Only fill within group — don't cross group boundaries (different products)
        for grp in groups:
            _fill_window(grp,
                         get_fn=lambda it: it.qui_cach,
                         set_fn=lambda it, v: setattr(it, 'qui_cach', v))

        # ── don_gia_co_vat: recompute after vat_pct is filled ────────────────
        for item in items:
            if item.don_gia is not None and item.vat_pct is not None and item.don_gia_co_vat is None:
                item.don_gia_co_vat = round(item.don_gia * (1 + item.vat_pct / 100), 0)

    @staticmethod
    def _post_process(doc: PriceDocument, skip_field_inheritance: bool = False) -> PriceDocument:
        """Fill missing fields with smart defaults — dvt, nhom_sp, VAT, qui_cach.

        skip_field_inheritance: set True when _inherit_sparse_fields was already applied
          per-page (scanned PDF flow) to prevent cross-page contamination.
        """
        # 1. Renumber stt globally 1..N
        for i, item in enumerate(doc.items, 1):
            item.stt = i

        # 1b. Propagate sparse fields (skip when already done per-page before merging)
        if not skip_field_inheritance:
            ExtractionAgent._inherit_sparse_fields(doc)

        # 1c. Forward-fill nhom_sp across the full merged document.
        _last_group: str | None = None
        for item in doc.items:
            if item.nhom_sp:
                _last_group = item.nhom_sp
            elif _last_group:
                item.nhom_sp = _last_group

        # 3b. Normalize DVT aliases to canonical Vietnamese units (rule-based, 0 API cost)
        dvt_changed = 0
        for item in doc.items:
            norm = _normalize_dvt(item.dvt or "")
            if norm != (item.dvt or ""):
                item.dvt = norm
                dvt_changed += 1
        if dvt_changed:
            logger.info(f"[PostProcess] DVT normalized: {dvt_changed} items")

        # 4. Infer dvt from product name when missing — key for cable/electrical items
        unit_map = [
            # Cable / wire (m — sold by length)
            ("dây cáp", "m"), ("dây dẫn", "m"), ("dây điện", "m"),
            ("cáp điện", "m"), ("cáp ngầm", "m"), ("cáp vặn xoắn", "m"),
            ("dây", "m"), ("cáp", "m"), ("cable", "m"), ("wire", "m"),
            ("ống ruột gà", "m"), ("ống luồn", "m"), ("ruột gà", "m"),
            # Pipes (m — sold by length)
            ("ống hdpe", "m"), ("ống pvc", "m"), ("ống pprc", "m"),
            ("ống thép", "m"), ("ống đồng", "m"), ("ống inox", "m"),
            ("ống", "m"), ("ruột", "m"),
            # Pipe fittings / valves (cái — sold individually)
            ("cút", "cái"), ("tê", "cái"), ("co ", "cái"),
            ("côn thu", "cái"), ("mặt bích", "cái"), ("bích", "cái"),
            ("van", "cái"), ("khớp nối", "cái"), ("khớp", "cái"),
            ("nối", "cái"), ("giảm", "cái"), ("đầu nối", "cái"),
            ("đầu bịt", "cái"), ("nắp bịt", "cái"), ("nút bịt", "cái"),
            ("chữ t", "cái"), ("chữ y", "cái"),
            # Electrical fittings (cái)
            ("hộp", "cái"), ("hộp nối", "cái"), ("máng cáp", "m"),
            ("thang cáp", "m"), ("máng", "m"),
            ("ổ cắm", "cái"), ("công tắc", "cái"), ("aptomat", "cái"),
            ("cầu dao", "cái"), ("rơ le", "cái"), ("relay", "cái"),
            ("cb ", "cái"), ("mcb", "cái"), ("mccb", "cái"), ("rccb", "cái"),
            ("biến áp", "cái"), ("tủ điện", "cái"), ("panel", "cái"),
            # Lighting (cái / bộ)
            ("bóng đèn", "cái"), ("đèn led", "bộ"), ("đèn tuýp", "bộ"),
            ("đèn", "bộ"), ("bóng", "cái"),
            # HVAC / mechanical (cái / bộ)
            ("máy lạnh", "bộ"), ("điều hòa", "bộ"), ("máy bơm", "cái"),
            ("quạt hút", "cái"), ("quạt thông gió", "cái"), ("quạt", "cái"),
            ("bình", "cái"), ("máy", "cái"), ("motor", "cái"),
            # Hardware / fasteners (bộ / cái)
            ("bu lông", "bộ"), ("bulong", "bộ"), ("đai ốc", "cái"),
            ("vít", "cái"), ("vis", "cái"), ("đinh", "cái"),
            ("gioăng", "cái"), ("ron", "cái"), ("seal", "cái"),
            ("giá đỡ", "cái"), ("ty ren", "m"), ("ty", "m"),
            # Sheets / boards (tấm)
            ("tấm", "tấm"), ("bản", "tấm"), ("miếng", "tấm"), ("sheet", "tấm"),
            # Rolls (cuộn)
            ("cuộn", "cuộn"), ("roll", "cuộn"), ("băng keo", "cuộn"),
            # Generic equipment (cái)
            ("thiết bị", "cái"), ("dụng cụ", "cái"),
        ]
        for item in doc.items:
            if item.dvt is None and item.ten_sp:
                name_lower = item.ten_sp.lower()
                for kw, inferred_unit in unit_map:
                    if kw in name_lower:
                        item.dvt = inferred_unit
                        break

        # 4b. Final fallback — DVT is mandatory; default to "cái" if still missing
        dvt_fallback = 0
        for item in doc.items:
            if item.dvt is None:
                item.dvt = "cái"
                dvt_fallback += 1
        if dvt_fallback:
            logger.info(f"[PostProcess] DVT fallback 'cái': {dvt_fallback} items")

        # 4b. Resolve price fields: ensure don_gia and don_gia_co_vat are both populated
        for item in doc.items:
            # Case A: have don_gia + vat_pct → compute don_gia_co_vat
            if item.don_gia is not None and item.vat_pct is not None and item.don_gia_co_vat is None:
                item.don_gia_co_vat = round(item.don_gia * (1 + item.vat_pct / 100), 0)
            # Case B: have don_gia_co_vat + vat_pct but no don_gia → compute don_gia
            elif item.don_gia is None and item.don_gia_co_vat is not None and item.vat_pct is not None and item.vat_pct > 0:
                item.don_gia = round(item.don_gia_co_vat / (1 + item.vat_pct / 100), 0)

        # 4c. Try to extract qui_cach from ten_sp when Gemini left it null.
        # Multi-domain regex: pipes/valves, cables/electrical, HVAC/mechanical
        import re as _re
        _QC_RE = _re.compile(
            r'('
            # 1. Pipes & valves: DN110×DN90 PN10, DN63, DN110xDN90, DN110-90, Phi 34, Ø20
            r'DN\s*\d+(?:\s*[×xX/\-*]\s*(?:DN\s*)?\d+)?(?:\s*PN\s*\d+)?'
            r'|PN\s*\d+'
            r'|(?:Phi|Φ|Ø|D)\s*\d+(?:\.\d+)?'
            # 2. Cables & electrical: 3x6+1x4mm², 3×6mm2, 0.6/1kV, 1.5kW, 220V, 50A
            r'|\d+\s*[×xX*]\s*[\d.]+(?:\s*\+\s*\d+\s*[×xX*]\s*[\d.]+)*\s*mm(?:²|2)?'
            r'|[\d.]+(?:/[\d.]+)?\s*kV'
            r'|\b\d+(?:\.\d+)?\s*(?:W|kW|V|A|mA|Hz)\b'
            # 3. HVAC & mechanical: 9000BTU, 2.0HP, 5ly, M16×1.5, 40×80×1.5mm
            r'|\b\d+(?:\.\d+)?\s*(?:BTU|HP|ly)\b'
            r'|\b\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?(?:\s*[×xX*]\s*\d+(?:\.\d+)?)?\s*(?:mm|cm)?\b'
            r'|\bM\d+(?:\s*[×xX*]\s*\d+(?:\.\d+)?)?(?:RC)?\b'
            r'|\d+\s*mm(?:²|2)'                     # 16mm² / 16mm2 standalone
            r')',
            _re.IGNORECASE,
        )
        for item in doc.items:
            if not item.qui_cach:
                # Try ten_sp first, then ghi_chu
                source = item.ten_sp or ""
                if not source and item.ghi_chu:
                    source = item.ghi_chu
                elif item.ghi_chu:
                    source = f"{item.ten_sp} {item.ghi_chu}"
                # findall collects ALL spec tokens (e.g. "PN10" + "DN63" → "PN10 DN63")
                matches = _QC_RE.findall(source)
                if matches:
                    item.qui_cach = " ".join(tok.strip() for tok in matches if tok.strip())

        # 4c2. Extract ma_sp from ten_sp and clean brand code from ten_sp.
        # Handles tables where model code is fused into product name:
        #   "RỌ ĐỒNG MBV PN10" → ten_sp="RỌ ĐỒNG", ma_sp="MBV PN10"
        # Also detects Gemini forward-fill errors (wrong brand in ma_sp vs ten_sp).
        _MA_SP_RE = _re.compile(
            r'\s+([A-Z]{2,8}(?:-[A-Z0-9]+)?\s+(?:PN|DN)\s*\d+)\s*$',
            _re.IGNORECASE
        )
        for item in doc.items:
            if not item.ten_sp:
                continue
            m = _MA_SP_RE.search(item.ten_sp)
            if not m:
                continue
            extracted_code = m.group(1).strip()
            cleaned_ten_sp = item.ten_sp[:m.start()].strip()
            if not cleaned_ten_sp:
                continue  # don't leave ten_sp empty
            item.ten_sp = cleaned_ten_sp
            if item.ma_sp is None:
                item.ma_sp = extracted_code
            else:
                # Detect Gemini forward-fill error: brand in ten_sp differs from ma_sp brand
                # e.g. ten_sp="…TURA PN10" but ma_sp="DALING-XK PN10" → replace
                ex_brand = extracted_code.split()[0].upper()
                cur_brand = item.ma_sp.split()[0].upper().split('-')[0]
                if ex_brand != cur_brand:
                    item.ma_sp = extracted_code

        # 4c3. Back-calculate vat_pct from don_gia + don_gia_co_vat when null.
        # Snaps to known Vietnamese VAT rates: 0, 5, 8, 10 (within ±0.5%).
        _KNOWN_VAT = (0.0, 5.0, 8.0, 10.0)
        for item in doc.items:
            if item.vat_pct is None and item.don_gia and item.don_gia_co_vat and item.don_gia > 0:
                raw_pct = (item.don_gia_co_vat / item.don_gia - 1) * 100
                for known in _KNOWN_VAT:
                    if abs(raw_pct - known) < 0.5:
                        item.vat_pct = known
                        break

        # 4d. Filter out items with no price (completely useless) or no name.
        # qui_cach and dvt are desired but NOT hard-filtered — lower confidence instead.
        before = len(doc.items)
        valid = []
        for item in doc.items:
            has_price = item.don_gia is not None or item.don_gia_co_vat is not None
            if not has_price:
                continue  # drop: price is absolutely required
            if not item.dvt:
                item.confidence = min(item.confidence, 0.5)
            if not item.qui_cach:
                item.confidence = min(item.confidence, 0.6)
            valid.append(item)
        doc.items = valid
        dropped = before - len(doc.items)
        if dropped:
            logger.warning(f"Dropped {dropped} items with no price")

        # 4e. Review null fields — fill page-break gaps, leave genuinely absent fields
        ExtractionAgent._review_null_fields(doc)

        # 4f. Confidence update after null review (fields may now be filled)
        for item in doc.items:
            if item.dvt:
                item.confidence = max(item.confidence, 0.5)
            if item.qui_cach:
                item.confidence = max(item.confidence, 0.6)

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

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Response may be truncated — attempt to recover complete items
                data = _recover_partial_json(text)

            # Parse items
            items: list[PriceItem] = []
            for item_data in data.get("items", []):
                try:
                    raw_ten_sp = item_data.get("ten_sp", "")
                    # Skip junk rows where ten_sp is 'nan' or empty
                    if str(raw_ten_sp).strip().lower() in ("nan", "none", ""):
                        continue
                    item = PriceItem(
                        stt=item_data.get("stt"),
                        nhom_sp=item_data.get("nhom_sp"),
                        ma_sp=item_data.get("ma_sp"),
                        ten_sp=raw_ten_sp,
                        qui_cach=item_data.get("qui_cach") or None,
                        dvt=item_data.get("dvt"),
                        don_gia=_safe_float(item_data.get("don_gia")),
                        don_gia_co_vat=_safe_float(item_data.get("don_gia_co_vat")),
                        dvt_2=item_data.get("dvt_2"),
                        don_gia_2=_safe_float(item_data.get("don_gia_2")),
                        vat_pct=_safe_float(item_data.get("vat_pct")),
                        ghi_chu=item_data.get("ghi_chu"),
                        confidence=float(item_data.get("confidence", 0.8)),
                    )
                    if item.ten_sp:
                        items.append(item)
                except Exception as e:
                    logger.warning(f"Skipping malformed item: {e}")

            doc = PriceDocument(
                gia_da_bao_gom_vat=data.get("gia_da_bao_gom_vat", False),
                nha_cung_cap=data.get("nha_cung_cap"),
                dia_chi=data.get("dia_chi"),
                dien_thoai=data.get("dien_thoai"),
                email=data.get("email"),
                ngay_hieu_luc=data.get("ngay_hieu_luc"),
                don_vi_tien=data.get("don_vi_tien", "VND"),
                items=items,
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


def _get_meta_val(meta, *keys):
    """Get a value from metadata that may be a dict or an object."""
    for key in keys:
        if isinstance(meta, dict):
            val = meta.get(key)
        else:
            val = getattr(meta, key, None)
        if val is not None:
            return val
    return None


def _is_truncated(response) -> bool:
    """Return True when the LLM stopped because it hit the output token limit."""
    try:
        rmeta = getattr(response, "response_metadata", {}) or {}
        finish = str(_get_meta_val(rmeta, "finish_reason", "stop_reason") or "").upper()
        if finish in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS", "LENGTH"):
            return True
        # Secondary signal: output_tokens close to the configured max
        meta = getattr(response, "usage_metadata", None)
        if meta:
            out_tok = _get_meta_val(meta, "output_tokens", "candidates_token_count") or 0
            if out_tok >= 64000:   # within 2% of 65536
                return True
    except Exception:
        pass
    return False


def _log_token_usage(response, label: str = "") -> None:
    """Log Gemini token usage and warn when output is truncated."""
    try:
        meta = getattr(response, "usage_metadata", None)
        if meta:
            in_tok  = _get_meta_val(meta, "input_tokens",  "prompt_token_count")
            out_tok = _get_meta_val(meta, "output_tokens", "candidates_token_count")
            total   = _get_meta_val(meta, "total_tokens",  "total_token_count")
            logger.info(f"[Token] {label} | in={in_tok} out={out_tok} total={total}")
            if out_tok and out_tok >= 65000:
                logger.warning(
                    f"[Token] ⚠️ Output gần limit ({out_tok}/65536) — "
                    "có thể bị truncate, tăng MAX_ROWS_PER_CHUNK hoặc kiểm tra lại kết quả"
                )
        rmeta = getattr(response, "response_metadata", {}) or {}
        finish = _get_meta_val(rmeta, "finish_reason", "stop_reason")
        if finish and str(finish).upper() in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS", "LENGTH"):
            logger.warning(
                f"[Token] ⚠️ finish_reason={finish} cho '{label}' — "
                "OUTPUT BỊ TRUNCATE, một số items có thể bị mất!"
            )
    except Exception:
        pass  # logging failure must never affect extraction


def _recover_partial_json(text: str) -> dict:
    """Attempt to recover usable data from a truncated JSON response.

    When Gemini hits its output token limit mid-stream the JSON is incomplete.
    Strategy: find the last well-formed item object, close the array/object, and
    parse the resulting (shorter but valid) JSON.  Raises ValueError on failure
    so the caller can fall through to the empty-doc fallback.
    """
    # Find the outermost "{" to locate the root object
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    # Find the end of the last COMPLETE item: look backwards for '}' that closes
    # an item dict (preceded by a value, not by '{'). We find the last '}' that
    # appears BEFORE the truncated region, then close the array and root object.
    items_start = text.find('"items"', start)
    if items_start < 0:
        raise ValueError("No items array found")

    # Try progressively shorter texts until json.loads succeeds
    # Walk backwards from the end, looking for '}'
    tail = text.rstrip()
    for _ in range(200):  # max 200 attempts
        last_brace = tail.rfind("}")
        if last_brace < 0:
            break
        candidate = tail[:last_brace + 1]
        # Close any open array and the root object
        for suffix in ("", "]}", "\n  ]\n}"):
            try:
                return json.loads(candidate + suffix)
            except json.JSONDecodeError:
                pass
        tail = tail[:last_brace]  # shrink and retry

    raise ValueError("Cannot recover partial JSON")


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
