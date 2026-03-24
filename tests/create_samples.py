"""Generate sample PDF files for testing without needing real documents."""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = BASE_DIR / "tests" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_soft_pdf():
    """Create a simple soft PDF with a price table."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm

        doc = SimpleDocTemplate(str(SAMPLES_DIR / "soft_baogia.pdf"), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>CÔNG TY TNHH CÔNG NGHỆ ABC</b>", styles['Title']))
        story.append(Paragraph("123 Nguyễn Huệ, Q.1, TP.HCM | ĐT: 028-1234-5678", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("<b>BÁO GIÁ SỐ: BG-2024-0315</b>", styles['Heading2']))
        story.append(Paragraph("Ngày: 15/03/2024 | Khách hàng: Công ty XYZ", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))

        data = [
            ['STT', 'Mã SP', 'Tên sản phẩm', 'ĐVT', 'SL', 'Đơn giá', 'Thành tiền'],
            ['1', 'LT-001', 'Laptop Dell XPS 15', 'Cái', '2', '35,000,000', '70,000,000'],
            ['2', 'MH-002', 'Màn hình Samsung 27"', 'Cái', '5', '8,500,000', '42,500,000'],
            ['3', 'BPH-003', 'Bàn phím cơ Keychron K2', 'Cái', '10', '2,200,000', '22,000,000'],
            ['4', 'CHP-004', 'Chuột Logitech MX Master 3', 'Cái', '10', '1,800,000', '18,000,000'],
            ['5', 'USB-005', 'USB Hub 7 Port Type-C', 'Cái', '8', '450,000', '3,600,000'],
            ['', '', '', '', '', 'Tổng chưa VAT:', '156,100,000'],
            ['', '', '', '', '', 'Thuế VAT (10%):', '15,610,000'],
            ['', '', '', '', '', 'TỔNG CỘNG:', '171,710,000'],
        ]

        table = Table(data, colWidths=[1.2*cm, 2*cm, 5.5*cm, 1.5*cm, 1.2*cm, 3*cm, 3*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a56db')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-6), 0.5, colors.grey),
            ('ALIGN', (4,1), (-1,-1), 'RIGHT'),
            ('BACKGROUND', (0,-3), (-1,-1), colors.HexColor('#f0f7ff')),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,-1), (-1,-1), 10),
        ]))
        story.append(table)
        doc.build(story)
        print(f"✅ Created: {SAMPLES_DIR}/soft_baogia.pdf")
    except ImportError:
        print("⚠️ reportlab not installed, creating text-based test instead.")
        _create_fallback_txt()


def create_sample_image():
    """Create a simple price table image for testing OCR."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        width, height = 1200, 800
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, width, 80], fill=(26, 86, 219))
        draw.text((30, 20), "BẢNG GIÁ VẬT LIỆU XÂY DỰNG - THÁNG 03/2024", fill=(255,255,255))

        # Table headers
        headers = ['STT', 'Tên hàng hóa', 'ĐVT', 'Đơn giá (đ)']
        x_positions = [30, 100, 600, 750]
        for i, (h, x) in enumerate(zip(headers, x_positions)):
            draw.text((x, 95), h, fill=(0, 0, 0))
        draw.line([(20, 115), (width-20, 115)], fill=(0,0,0), width=2)

        # Data rows
        rows = [
            ('1', 'Xi măng Hà Tiên PCB40 (50kg/bao)', 'Bao', '95,000'),
            ('2', 'Cát xây (1m3)', 'm3', '280,000'),
            ('3', 'Đá 1x2 (1m3)', 'm3', '420,000'),
            ('4', 'Gạch ống 4 lỗ (1,000 viên)', '1000v', '2,500,000'),
            ('5', 'Sắt phi 12 (1 tấn)', 'Tấn', '15,800,000'),
            ('6', 'Tôn lạnh dày 0.3mm (m2)', 'm2', '75,000'),
            ('7', 'Ngói xi măng màu (viên)', 'Viên', '4,500'),
            ('8', 'Cửa nhôm kính 1 cánh (1m x 2m)', 'Bộ', '3,200,000'),
        ]

        y = 125
        for i, row in enumerate(rows):
            bg = (240, 247, 255) if i % 2 == 0 else (255, 255, 255)
            draw.rectangle([20, y-3, width-20, y+23], fill=bg)
            for val, x in zip(row, x_positions):
                draw.text((x, y), val, fill=(20, 20, 20))
            y += 30

        img_path = SAMPLES_DIR / "image_banggia.jpg"
        img.save(str(img_path), "JPEG", quality=95)
        print(f"✅ Created: {img_path}")
    except ImportError:
        print("⚠️ Pillow not available for image sample generation.")


def _create_fallback_txt():
    txt_path = SAMPLES_DIR / "sample_data.txt"
    txt_path.write_text("""BẢNG BÁO GIÁ - TEST DATA
Số BG: BG-2024-001
Ngày: 15/03/2024
Nhà cung cấp: Công ty Test ABC

STT | Mã SP   | Tên sản phẩm          | ĐVT | SL | Đơn giá     | Thành tiền
1   | LT-001  | Laptop Dell XPS 15    | Cái | 2  | 35,000,000  | 70,000,000
2   | MH-002  | Màn hình Samsung 27"  | Cái | 5  | 8,500,000   | 42,500,000
3   | BPH-003 | Bàn phím cơ K2        | Cái | 10 | 2,200,000   | 22,000,000

Tổng chưa VAT: 134,500,000
VAT (10%):      13,450,000
TỔNG CỘNG:    147,950,000
""", encoding="utf-8")
    print(f"✅ Created fallback: {txt_path}")


if __name__ == "__main__":
    print("🔨 Tạo sample files cho testing...")
    create_sample_soft_pdf()
    create_sample_image()
    print("✅ Hoàn thành! Sample files trong tests/samples/")
