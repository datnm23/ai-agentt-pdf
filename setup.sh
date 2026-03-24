#!/bin/bash
# ──────────────────────────────────────────────────────────────
# AI Agent PDF Báo Giá — Setup & Run Script
# ──────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

echo "🚀 AI Agent PDF Báo Giá"
echo "========================"

# ── 1. Create virtual environment if needed ───────────────────
if [ ! -d "venv" ]; then
  echo "📦 Tạo virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# ── 2. Install/upgrade dependencies ──────────────────────────
echo "📥 Cài đặt dependencies..."
pip install --upgrade pip --quiet

# Install in stages to handle heavy packages
pip install fastapi uvicorn[standard] python-multipart aiofiles --quiet
pip install langchain langchain-google-genai langchain-core google-generativeai --quiet
pip install pdfplumber pypdf pdf2image Pillow --quiet
pip install easyocr --quiet
pip install opencv-python-headless numpy scikit-image scipy --quiet
pip install pandas openpyxl pydantic --quiet
pip install sqlalchemy aiosqlite --quiet
pip install python-dotenv python-magic filetype loguru --quiet

# Try PaddleOCR (optional, heavy)
pip install paddlepaddle paddleocr --quiet 2>/dev/null || echo "⚠️  PaddleOCR skip (installs separately)"

# ── 3. Create runtime directories ────────────────────────────
mkdir -p uploads results tests/samples

# ── 4. Create .env if not exists ──────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "⚠️  .env created from .env.example — verify your GOOGLE_API_KEY"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "▶  To start the server:"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload --port 8000"
echo ""
echo "🌐 Then open: http://localhost:8000"
