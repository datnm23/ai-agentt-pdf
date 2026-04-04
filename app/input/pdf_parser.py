"""PDF-specific parser — handles soft PDFs with text layers using pdfplumber."""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from loguru import logger

# ── Canonical column alias map ──────────────────────────────────────────────
# Maps non-standard Vietnamese column names → canonical names that Gemini
# recognises reliably. Only include entries that are genuinely ambiguous;
# Gemini handles common aliases (Tên SP, Đơn giá…) by itself.
_COLUMN_ALIAS_MAP: dict[str, str] = {
    # QUY CÁCH — size/spec column variants
    'TIẾT DIỆN': 'QUY CÁCH', 'KÍCH THƯỚC': 'QUY CÁCH',
    'SPEC': 'QUY CÁCH', 'SIZE': 'QUY CÁCH', 'CÁCH': 'QUY CÁCH',
    # MÃ SP — product code column variants
    'KÝ HIỆU': 'MÃ SP', 'KY HIEU': 'MÃ SP', 'CODE': 'MÃ SP',
    'PART NO': 'MÃ SP', 'SKU': 'MÃ SP', 'MODEL': 'MÃ SP',
    'MÃ HÀNG': 'MÃ SP', 'MA HANG': 'MÃ SP',
    # ĐƠN GIÁ variants → normalise to one label
    'GIÁ CHƯA VAT': 'ĐƠN GIÁ', 'ĐƠN GIÁ CHƯA VAT': 'ĐƠN GIÁ',
    'GIÁ (VNĐ)': 'ĐƠN GIÁ', 'ĐƠN GIÁ (VNĐ)': 'ĐƠN GIÁ',
    # ĐƠN GIÁ CÓ VAT
    'GIÁ CÓ VAT': 'ĐƠN GIÁ CÓ VAT', 'GIÁ SAU VAT': 'ĐƠN GIÁ CÓ VAT',
    'ĐƠN GIÁ SAU VAT': 'ĐƠN GIÁ CÓ VAT',
}

# Regex: column names that represent size/dimension variants (matrix tables)
_SIZE_COL_RE = re.compile(
    r'^(DN|PN|D|Φ|phi)[\s\-]?\d+$'   # DN15, PN20, D32
    r'|^\d+[\s×xX]\d+'                # 2×1.5, 4×6
    r'|^\d+\s*mm²?$'                  # 16mm², 25mm
    r'|^(loại|cỡ|size)\s*\d+$',       # loại 1, cỡ 2
    re.IGNORECASE
)

# Regex: column names that look like "price per unit" (multi-unit price tables)
_PRICE_UNIT_COL_RE = re.compile(
    r'(VNĐ|VND|đ)\s*/\s*\w+'          # VNĐ/m, VND/kg
    r'|giá\s*[/\(]'                    # Giá/m, Giá (m)
    r'|/\s*(m|kg|cái|cuộn|bộ|hộp|thùng)\b',  # /m, /kg
    re.IGNORECASE
)

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

    @staticmethod
    def _extract_page_section_header(page_text: str) -> Optional[str]:
        """Return the product-section heading from a page's text.

        In Vietnamese price catalogs the page layout is consistent:
          Line 1 — company tagline (always the same, skip)
          Line 2 — subtitle in mixed case (skip)
          Line 3 — specific section title, e.g. "VAN VÀ PHỤ KIỆN PP-R MINH HÒA"
                   OR column-header text when the page is a continuation (no new section)

        We inspect lines 3–5 (0-based: indices 2–4) and return the first line that
        is ALL-CAPS and is not a generic / column-header line.
        """
        _EXACT_SKIP = {'VAN VÒI MINH HÒA', 'SẢN XUẤT TẠI VIỆT NAM',
                       'NHÃN HIỆU FUZHOU – SẢN XUẤT TRUNG QUỐC'}
        _CONTAINS_SKIP = (
            'VAN VÒI MINH HÒA – ',   # company tagline
            'THƯƠNG HIỆU UY TÍN',
            'QUY GIÁ', 'GIÁ CÓ', 'GIÁ CHƯA VAT',
            'ĐÓNG GÓI', 'STT',
            'VANVOIMINHHOA',
        )
        lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
        # Only examine lines 3–5 (after company name + subtitle)
        for line in lines[2:5]:
            if not line or len(line) < 6:
                continue
            if line in _EXACT_SKIP:
                continue
            if any(frag in line for frag in _CONTAINS_SKIP):
                continue
            alpha = [c for c in line if c.isalpha()]
            if len(alpha) < 5:
                continue
            if sum(c.isupper() for c in alpha) / len(alpha) >= 0.8:
                return line
        return None

    def extract_tables(self, pdf_path: Path) -> List[Tuple[Optional[str], pd.DataFrame]]:
        """Extract tabular data from soft PDF using pdfplumber.

        Returns a list of ``(section_header, dataframe)`` pairs so that callers
        can include the section context in the LLM prompt.
        """
        if not HAS_PDFPLUMBER:
            return []
        tables: List[Tuple[Optional[str], pd.DataFrame]] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    section_header = self._extract_page_section_header(page_text)
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        if tbl and len(tbl) > 1:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            df = df.fillna("").astype(str)
                            df = df[df.apply(lambda row: any(c.strip() for c in row), axis=1)]
                            if len(df) > 0:
                                tables.append((section_header, df))
                                logger.debug(
                                    f"Page {page_num}: extracted table with {len(df)} rows"
                                    + (f", section={section_header!r}" if section_header else "")
                                )
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

    @staticmethod
    def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty columns/rows caused by merged-cell expansion in pdfplumber.

        pdfplumber sometimes places cell data one column to the LEFT of its header
        column when merged-cell headers span multiple PDF columns. This method:
          1. Drops completely empty gap columns
          2. Detects (unnamed data col, named label col) adjacent pairs and merges them
          3. Absorbs subheader rows (e.g. "(ĐVT: VNĐ/Cái)") into column names
        """
        # ── Step 1: drop completely empty columns ───────────────────
        keep_idx = [
            i for i in range(len(df.columns))
            if df.iloc[:, i].astype(str).str.strip().str.len().gt(0).any()
        ]
        df = df.iloc[:, keep_idx].copy() if keep_idx else df.copy()

        # ── Step 2: drop entirely empty rows ────────────────────────
        df = df[df.apply(lambda row: any(str(c).strip() for c in row), axis=1)].copy()
        df = df.reset_index(drop=True)

        if len(df) == 0:
            return df

        # ── Step 3: fix off-by-one column alignment ─────────────────
        # Pattern: unnamed col[i] has real data rows, named col[i+1] is empty in
        # data rows but has a subheader in row 0.  Rename col[i] with col[i+1]'s
        # label + subheader, then drop col[i+1].
        cols = list(df.columns)
        new_names = cols.copy()
        drop_set: set[int] = set()

        for i in range(len(cols) - 1):
            curr_name = str(cols[i]).strip()
            next_name = str(cols[i + 1]).strip()
            if curr_name != '' or next_name == '':
                continue
            # curr col is unnamed, next col has a label
            curr_data = df.iloc[1:, i].astype(str).str.strip().str.len().gt(0)
            next_data = df.iloc[1:, i + 1].astype(str).str.strip().str.len().gt(0)
            if curr_data.any() and not next_data.any():
                # Data lives in the unnamed col; the label is in the next col
                sub = str(df.iloc[0, i + 1]).strip()
                label = next_name
                if sub and sub not in ('nan', '', label):
                    new_names[i] = f"{label} {sub}"
                else:
                    new_names[i] = label
                drop_set.add(i + 1)

        df.columns = new_names
        if drop_set:
            keep = [i for i in range(len(df.columns)) if i not in drop_set]
            df = df.iloc[:, keep].copy()

        # ── Step 3b: fix undetected header row ──────────────────────
        # Pattern: pdfplumber left the actual header row (containing "STT",
        # "TÊN - HÌNH ẢNH", etc.) as a data row because all/many column names
        # are '' (empty).  Detect this and promote it to column names.
        COLUMN_HEADER_KW = ('STT', 'TT', 'TÊN', 'TEN', 'MÃ', 'MA SP', 'CODE',
                            'DVT', 'ĐVT', 'HÌNH ẢNH', 'HANG HOA', 'SẢN PHẨM')
        empty_col_count = sum(1 for c in df.columns if str(c).strip() in ('', 'nan'))
        if empty_col_count > 0 and len(df) >= 2:
            # Check first two rows — some tables (e.g. 3-row merged headers) have the
            # STT/TÊN row at position 1 (when position 0 has partial header labels).
            for _hrow in range(min(2, len(df))):
                candidate_vals = [str(v).strip() for v in df.iloc[_hrow]]
                is_hidden_header = any(
                    any(kw in val.upper() for kw in COLUMN_HEADER_KW)
                    for val in candidate_vals
                )
                if is_hidden_header:
                    # Absorb any preceding partial-header rows into column names first
                    for _prior in range(_hrow):
                        prior_vals = [str(v).strip() for v in df.iloc[0]]
                        new_col_names = list(df.columns)
                        for i, (col, val) in enumerate(zip(df.columns, prior_vals)):
                            if val and val.upper() not in ('NAN', '') and str(col).strip() in ('', 'nan'):
                                new_col_names[i] = val
                            elif val and val.upper() not in ('NAN', ''):
                                # Append partial header to existing name (e.g. "GIÁ CÓ" + "VAT" → "GIÁ CÓ VAT")
                                new_col_names[i] = f"{str(col).strip()} {val}".strip()
                        df.columns = new_col_names
                        df = df.iloc[1:].reset_index(drop=True)
                        candidate_vals = [str(v).strip() for v in df.iloc[0]]
                    # Now promote the hidden header row
                    new_col_names = list(df.columns)
                    for i, (col, val) in enumerate(zip(df.columns, candidate_vals)):
                        if str(col).strip() in ('', 'nan') and val and val.upper() not in ('NAN', ''):
                            new_col_names[i] = val
                    df.columns = new_col_names
                    df = df.iloc[1:].reset_index(drop=True)
                    break

        # ── Step 4: absorb any remaining subheader row ───────────────
        # After the alignment fix the first row may still carry subheader text for
        # columns whose names didn't need fixing (e.g. the original named cols).
        # SUBHEADER_KW: only unit/format tokens that cannot appear in real product data.
        # 'DN' and 'PN' intentionally excluded — they appear in product sizes (DN 15, PN 20).
        if len(df) == 0:
            return df
        first_vals = [str(v).strip() for v in df.iloc[0]]
        SUBHEADER_KW = ('ĐVT', 'đvt', 'Unit', 'VNĐ', 'VND', 'CÁCH', 'Cái', 'mm²')
        is_subheader = any(
            kw in val for val in first_vals for kw in SUBHEADER_KW
        )
        if is_subheader:
            new_cols = []
            for col, sub in zip(df.columns, first_vals):
                col_s = str(col).strip()
                sub_s = sub
                if sub_s and sub_s not in ('nan', '') and sub_s != col_s:
                    new_cols.append(f"{col_s} {sub_s}".strip() if col_s else sub_s)
                else:
                    new_cols.append(col_s if col_s not in ('nan', '') else sub_s)
            df.columns = new_cols
            df = df.iloc[1:].reset_index(drop=True)

        # ── Step 5: forward-fill STT and TÊN SP for sub-rows ────────
        # Tables often have sub-rows (size variants, DN sizes) with blank TÊN/STT.
        # Fill down so Gemini sees the product name on every row.
        name_col = next(
            (c for c in df.columns if any(kw in str(c).upper()
             for kw in ('TÊN', 'TEN', 'SẢN PHẨM', 'SAN PHAM', 'HÀNG HÓA', 'HANG HOA'))),
            None
        )
        stt_col = next(
            (c for c in df.columns
             if str(c).upper().strip() in ('STT', 'TT', 'NO', 'SỐ TT', 'SO TT')),
            None
        )
        if name_col:
            df[name_col] = df[name_col].replace('', pd.NA).replace('nan', pd.NA).ffill()
            # Split "Tên SP\nKý hiệu" multi-line cells into separate KÝ HIỆU column
            # E.g. "VAN CỬA ĐỒNG\nMIHA-XK PN20" → ten_sp="VAN CỬA ĐỒNG", ky_hieu="MIHA-XK PN20"
            # Only split when the 2nd line looks like a model code (not a name continuation).
            # Model code pattern: alphanumeric+hyphens/dots, often with dash separators,
            # e.g. "MIHA-XK PN20", "CV 0.5R5-0.3" — NOT "ỐNG MỀM ĐỒNG MH" or "THÂN GANG NỐI".
            import re as _re
            _MODEL_CODE_RE = _re.compile(
                r'^[A-Z0-9]{2,}[-./][A-Z0-9]'   # starts with uppercase+digits + separator
                r'|^[A-Z]{2,}\d'                 # CX123, PN20 style
                r'|^\d+[xX×]\d'                  # 2x1.5, 4×6 sizes
                r'|^(DN|PN|D|Φ)\s*\d+',          # DN15, PN20
                _re.IGNORECASE
            )
            if 'KÝ HIỆU' not in [str(c).upper() for c in df.columns]:
                def _split_name_code(val):
                    s = str(val) if val is not None else ''
                    parts = [p.strip() for p in s.split('\n') if p.strip() and p.strip() != 'nan']
                    if len(parts) >= 2 and _MODEL_CODE_RE.match(parts[1]):
                        return parts[0], parts[1]
                    # Not a code — keep the full name (join back or take first part)
                    return s.replace('\n', ' ').strip(), None

                split_vals = df[name_col].apply(_split_name_code)
                names = split_vals.apply(lambda x: x[0])
                codes = split_vals.apply(lambda x: x[1])
                # Always clean up name column (replace \n even when no codes found)
                df[name_col] = names
                if codes.notna().any():
                    df.insert(df.columns.get_loc(name_col) + 1, 'KÝ HIỆU', codes)
                    # Forward-fill the code to sub-rows (same model applies to all sizes)
                    df['KÝ HIỆU'] = df['KÝ HIỆU'].replace('', pd.NA).ffill()
        if stt_col:
            df[stt_col] = df[stt_col].replace('', pd.NA).replace('nan', pd.NA).ffill()

        # ── Step 6: map ambiguous column names → canonical names ─────
        df = PDFParser._map_canonical_columns(df)

        # ── Step 7: melt matrix / multi-unit-price columns ───────────
        df = PDFParser._normalize_price_columns(df)

        return df

    @staticmethod
    def _map_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename non-standard column names to canonical Vietnamese names.

        Only renames entries in _COLUMN_ALIAS_MAP (case-insensitive, stripped).
        Helps Gemini reliably identify QUY CÁCH, ĐƠN GIÁ CÓ VAT, etc.
        """
        rename = {}
        for col in df.columns:
            key = str(col).upper().strip()
            # Exact match first
            if key in _COLUMN_ALIAS_MAP:
                rename[col] = _COLUMN_ALIAS_MAP[key]
                continue
            # Partial/prefix match for columns like 'GIÁ CHƯA VAT (ĐVT: VNĐ/Cái)'
            for alias, canonical in _COLUMN_ALIAS_MAP.items():
                if key.startswith(alias):
                    rename[col] = canonical
                    break
        if rename:
            logger.debug(f"Canonical column rename: {rename}")
            df = df.rename(columns=rename)
        return df

    @staticmethod
    def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Melt matrix tables (size columns like DN15/DN20) or multi-unit-price
        tables (VNĐ/m | VNĐ/kg) into a canonical tall format.

        Each (product × variant) becomes one row with:
          - GHI CHÚ = variant name (DN15, VNĐ/kg, …)
          - ĐƠN GIÁ = price for that variant

        Returns df unchanged if no size/price-unit columns are detected.
        """
        # If the table already has a QUY CÁCH / TIẾT DIỆN column, it's already tall — skip melt
        has_qui_cach_col = any(
            str(c).upper().strip() in ('QUY CÁCH', 'QUY CACH', 'TIẾT DIỆN', 'KÍCH THƯỚC')
            for c in df.columns
        )
        if has_qui_cach_col:
            return df

        # Prefer size-matrix pattern (DN15, PN20, 2×1.5...)
        size_cols = [c for c in df.columns if _SIZE_COL_RE.match(str(c).strip())]

        # Fall back to multi-unit-price pattern (VNĐ/m | VNĐ/kg)
        # Exclude: GIÁ CHƯA VAT / GIÁ CÓ VAT (pre/post-VAT columns, not unit variants)
        # Exclude: ĐÓNG GÓI and packaging columns
        _SKIP_KW = ('CHƯA VAT', 'CÓ VAT', 'SAU VAT', 'TRƯỚC VAT', 'ĐÓNG GÓI', 'PACKAGING')
        if not size_cols:
            size_cols = [
                c for c in df.columns
                if _PRICE_UNIT_COL_RE.search(str(c).strip())
                and str(c).strip().upper() not in ('TÊN SẢN PHẨM', 'MÃ SP', 'QUY CÁCH')
                and not any(kw in str(c).upper() for kw in _SKIP_KW)
            ]
        if not size_cols:
            return df

        anchor_cols = [c for c in df.columns if c not in size_cols]
        melted = df.melt(
            id_vars=anchor_cols,
            value_vars=size_cols,
            var_name='QUY CÁCH',   # size label → qui_cach, not ghi_chu
            value_name='ĐƠN GIÁ',
        )
        # Drop rows where the size-variant price is empty
        melted = melted[melted['ĐƠN GIÁ'].astype(str).str.strip().str.len() > 0].copy()
        melted = melted.reset_index(drop=True)
        logger.debug(
            f"Matrix melt: {len(size_cols)} price-cols × {len(df)} rows → {len(melted)} rows"
        )
        return melted

    @staticmethod
    def _table_quality_score(df: pd.DataFrame) -> float:
        """Score 0.0–1.0: average of named-column ratio and filled-row ratio."""
        if df.empty or len(df.columns) == 0:
            return 0.0
        named_cols = sum(1 for c in df.columns
                         if str(c).strip() not in ('', 'nan', 'None'))
        col_score = named_cols / len(df.columns)
        filled_rows = df.apply(lambda r: any(str(c).strip() for c in r), axis=1).mean()
        return col_score * 0.5 + float(filled_rows) * 0.5

    def extract_tables_best(
        self, pdf_path: Path
    ) -> List[Tuple[Optional[str], pd.DataFrame]]:
        """Extract tables using pdfplumber; fall back to Camelot if quality is low."""
        tables = self.extract_tables(pdf_path)
        if not tables:
            logger.info(f"pdfplumber found no tables, trying Camelot for {pdf_path.name}")
            return [(None, df) for df in self.extract_tables_camelot(pdf_path)]

        dfs = [df for _, df in tables]
        avg_quality = sum(self._table_quality_score(df) for df in dfs) / len(dfs)
        logger.debug(f"pdfplumber table quality={avg_quality:.2f} for {pdf_path.name}")
        if avg_quality < 0.5:
            logger.info(f"pdfplumber quality={avg_quality:.2f} < 0.5, trying Camelot fallback")
            camelot_dfs = self.extract_tables_camelot(pdf_path)
            if camelot_dfs:
                camelot_quality = sum(self._table_quality_score(df)
                                      for df in camelot_dfs) / len(camelot_dfs)
                logger.info(f"Camelot quality={camelot_quality:.2f}")
                if camelot_quality > avg_quality:
                    return [(None, df) for df in camelot_dfs]
        return tables

    def tables_to_text(
        self, tables: List[Tuple[Optional[str], pd.DataFrame]]
    ) -> str:
        """Convert extracted DataFrames to readable text for LLM.

        Accepts the ``(section_header, dataframe)`` pairs returned by
        ``extract_tables`` / ``extract_tables_best``.  The section header
        (e.g. "VAN VÀ PHỤ KIỆN PP-R MINH HÒA") is prepended before the first
        table in each section so Gemini can use it as ``nhom_sp`` context.
        Continuation tables (no new section header) inherit the current section,
        so the heading only appears once per section group.

        Cleans up merged-cell artefacts, renders as Markdown tables (requires
        tabulate), and falls back to to_string() when tabulate is unavailable.
        """
        parts = []
        current_section: Optional[str] = None
        for i, item in enumerate(tables):
            section_header, df = item
            df = self._clean_table(df)
            # Emit section header whenever a new named section begins
            if section_header and section_header != current_section:
                current_section = section_header
                parts.append(f"\n### NHÓM SẢN PHẨM: {current_section}")
            parts.append(f"=== Bảng {i+1} ===")
            try:
                parts.append(df.to_markdown(index=False))
            except Exception:
                parts.append(df.to_string(index=False))
            parts.append("")
        return "\n".join(parts)
