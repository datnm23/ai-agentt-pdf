"""PDF-specific parser — handles soft PDFs with text layers using pdfplumber."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from loguru import logger

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class PDFParser:
    """
    Extract text and tables from native (soft) PDFs.
    Falls back to image conversion for scanned PDFs.
    """

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a soft PDF."""
        if not HAS_PDFPLUMBER:
            return ""
        try:
            texts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        texts.append(text)
            result = "\n\n".join(texts)
            logger.info(f"pdfplumber extracted {len(result)} chars from {pdf_path.name}")
            return result
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed: {e}")
            return ""

    def extract_tables(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tabular data from soft PDF using pdfplumber."""
        if not HAS_PDFPLUMBER:
            return []
        tables = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        if tbl and len(tbl) > 1:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            df = df.fillna("").astype(str)
                            df = df[df.apply(lambda row: any(c.strip() for c in row), axis=1)]
                            if len(df) > 0:
                                tables.append(df)
                                logger.debug(f"Page {page_num}: extracted table with {len(df)} rows")
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {e}")
        return tables

    def extract_tables_camelot(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Try camelot for complex tables (lattice + stream mode)."""
        try:
            import camelot
            tables = []
            # Try lattice mode first (for bordered tables)
            try:
                tlist = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
                for t in tlist:
                    if t.accuracy > 60:
                        tables.append(t.df)
                        logger.debug(f"Camelot lattice: accuracy={t.accuracy:.1f}%")
            except Exception:
                pass

            # If no tables found, try stream mode (for borderless tables)
            if not tables:
                try:
                    tlist = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
                    for t in tlist:
                        if t.accuracy > 50:
                            tables.append(t.df)
                            logger.debug(f"Camelot stream: accuracy={t.accuracy:.1f}%")
                except Exception:
                    pass

            return tables
        except ImportError:
            logger.debug("camelot not installed, skipping")
            return []

    def pdf_to_images(self, pdf_path: Path, output_dir: Path, dpi: int = 200) -> List[Path]:
        """Convert PDF pages to images for OCR processing."""
        if not HAS_PDF2IMAGE:
            logger.warning("pdf2image not installed")
            return []
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            images = convert_from_path(str(pdf_path), dpi=dpi)
            paths = []
            for i, img in enumerate(images):
                img_path = output_dir / f"page_{i+1:03d}.jpg"
                img.save(str(img_path), "JPEG", quality=95)
                paths.append(img_path)
            logger.info(f"Converted {len(paths)} PDF pages to images")
            return paths
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return []

    def tables_to_text(self, tables: List[pd.DataFrame]) -> str:
        """Convert extracted DataFrames to readable text for LLM."""
        parts = []
        for i, df in enumerate(tables):
            parts.append(f"=== Bảng {i+1} ===")
            parts.append(df.to_string(index=False))
            parts.append("")
        return "\n".join(parts)
