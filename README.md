# AI Agent PDF Báo Giá 🤖📄

Hệ thống AI Agent tự động trích xuất dữ liệu từ **mọi loại** file báo giá/bảng giá.

## Loại tài liệu hỗ trợ

| Loại | Xử lý |
|------|-------|
| ✅ PDF mềm (text layer) | pdfplumber + camelot |
| ✅ PDF scan | PDF → ảnh → OCR |
| ✅ Ảnh scan phẳng | OpenCV + OCR |
| ✅ Ảnh chụp điện thoại | Deskew + OCR |
| ✅ Ảnh lóa sáng | CLAHE + Inpainting |
| ✅ Ảnh nghiêng/vẹo | Perspective transform |

## Cài đặt nhanh

```bash
chmod +x setup.sh && ./setup.sh
```

## Chạy server

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Mở trình duyệt: **http://localhost:8000**

## Cấu trúc dự án

```
ai-agentt-pdf/
├── app/
│   ├── input/          # Document detector + PDF parser
│   ├── preprocessing/  # OpenCV image pipeline
│   ├── ocr/            # PaddleOCR / EasyOCR / Gemini Vision
│   ├── agents/         # LangChain + Gemini Flash extraction
│   ├── api/            # FastAPI routes
│   ├── db/             # SQLite database
│   ├── services/       # Job orchestration + Export
│   └── models/         # Pydantic schemas
├── frontend/           # HTML + CSS + JS dashboard
├── uploads/            # Uploaded files (auto-created)
├── results/            # JSON results + Excel exports
├── tests/              # Test samples
└── .env                # API keys & config
```

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| POST | `/api/upload` | Upload file |
| GET | `/api/jobs/{id}/status` | Polling trạng thái |
| GET | `/api/results/{id}` | Lấy kết quả JSON |
| GET | `/api/export/{id}?format=excel` | Export Excel/CSV/JSON |
| GET | `/api/history` | Lịch sử xử lý |
| DELETE | `/api/jobs/{id}` | Xóa job |

## Environment Variables

```env
GOOGLE_API_KEY_VISION=your-key-for-gemini-2.5-flash     # Vision: SCANNED_PDF, ảnh
GOOGLE_API_KEY_TEXT=your-key-for-gemini-2.5-flash-lite  # Text: SOFT_PDF, OCR output
GOOGLE_API_KEY=fallback-key-if-not-set-above
GEMINI_VISION_MODEL=gemini-2.5-flash
GEMINI_TEXT_MODEL=gemini-2.5-flash-lite
OCR_ENGINE=auto            # auto | paddleocr | easyocr | tesseract
CONFIDENCE_THRESHOLD=0.65
MAX_FILE_SIZE_MB=50
```
