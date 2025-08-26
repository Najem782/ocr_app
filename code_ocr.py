#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Invoice Extractor (OCR from PDF)
------------------------------------------
Uploads one or more PDF invoices, performs OCR, extracts key fields with regex,
normalizes them, and lets the user download a CSV.
"""
import os
import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

# -------------------------
# App Config & Sidebar
# -------------------------
st.set_page_config(page_title="Invoice Extractor (OCR)", page_icon="📄", layout="wide")
st.title("📄 Invoice Extractor (OCR from PDF)")
st.write(
    "Upload one or more scanned PDF invoices. The app will extract invoice number, total amount, date, and sender using OCR."
)

with st.sidebar:
    st.header("⚙️ Settings")
    dpi = st.slider("PDF render DPI", min_value=200, max_value=400, value=300, step=50)
    use_threshold = st.checkbox("Binarize (improves OCR on low-contrast scans)", value=True)
    show_ocr_text = st.checkbox("Show OCR text preview", value=False)
    tesseract_cmd = st.text_input(
        "Tesseract path (Windows, optional)",
        help="If Tesseract isn't on PATH, put its full path here, e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        value=os.environ.get("TESSERACT_CMD", ""),
    )
    poppler_path = st.text_input(
        "Poppler bin path (Windows, optional)",
        help="If Poppler isn't on PATH, set the folder containing poppler's bin, e.g. C:\\poppler-xx\\Library\\bin",
        value=os.environ.get("POPPLER_PATH", ""),
    )

if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# -------------------------
# OCR Helpers
# -------------------------
def preprocess_image(img: Image.Image, do_threshold: bool = True) -> Image.Image:
    # Convert to grayscale, optionally binarize, slightly sharpen
    img = ImageOps.grayscale(img)
    if do_threshold:
        img = ImageOps.autocontrast(img)
        # Simple global threshold (Otsu would be nicer, but keep dependencies minimal)
        img = img.point(lambda x: 255 if x > 160 else 0, mode="1").convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_from_pdf_bytes(pdf_bytes: BytesIO, dpi: int = 300, do_threshold: bool = True, poppler_path: str = "") -> Tuple[str, List[str]]:
    """
    Returns (full_text, per_page_texts)
    """
    convert_kwargs = {"dpi": dpi}
    if poppler_path:
        convert_kwargs["poppler_path"] = poppler_path

    images = convert_from_bytes(pdf_bytes.read(), **convert_kwargs)
    page_texts: List[str] = []
    for page_number, page in enumerate(images, start=1):
        page = preprocess_image(page, do_threshold=do_threshold)
        text = pytesseract.image_to_string(page, lang="eng", config="--psm 6 --oem 3")
        page_texts.append(text)
    full_text = "\n\n".join(page_texts)
    return full_text, page_texts

# -------------------------
# Extraction Helpers
# -------------------------
NUMBER_PATTERNS = [
    r"\bInvoice(?:\s*No\.?|#| Number)?\s*[:\-]?\s*([A-Z0-9\-\/]{5,})",
    r"\bDelivery Advice No\.\s*[:\-]?\s*(\d{6,})",
]
AMOUNT_PATTERNS = [
    r"\bTotal(?:\s*Amount)?\s*(?:USD|US\$|\$)?\s*[:\-]?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)",
    r"\bInvoice\s*(?:USD|US\$|\$)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)",
    r"\bGrand\s*Total\s*(?:USD|US\$|\$)?\s*[:\-]?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)",
]
DATE_PATTERNS = [
    r"\bDated\s*[:\-]?\s*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})",
    r"\bDate\s*[:\-]?\s*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})",
    r"\bInvoice\s*Date\s*[:\-]?\s*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})",
    r"\b(\d{4}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{1,2})\b",
]
SENDER_PATTERNS = [
    r"(?i)\b([A-Z][A-Za-z0-9&\.,\-\s]{2,}?\s+(?:LLC|L\.L\.C\.|Inc\.?|Incorporated|Ltd\.?|Limited|GmbH|S\.A\.|S\.A\.R\.L\.|Pte\.? Ltd\.?|FZE|FZ-LLC|BV|NV))\b",
    r"(?i)\b([A-Z][A-Za-z0-9&\.,\-\s]{3,}?\s+Emerging Markets FZE)\b",
]

def _first_match(patterns: List[str], text: str, flags: int = 0) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return ""

def normalize_amount(s: str) -> str:
    if not s:
        return ""
    # Remove thousands separator, ensure dot decimal
    s = s.replace(",", "")
    return s

def normalize_date(s: str) -> str:
    if not s:
        return ""
    # Try day-first and month-first; fall back to raw
    for dayfirst in (True, False):
        try:
            dt = pd.to_datetime(s, dayfirst=dayfirst, errors="raise")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return s  # return raw if parsing failed

def extract_invoice_data(text: str) -> Dict[str, str]:
    inv = _first_match(NUMBER_PATTERNS, text, flags=re.IGNORECASE)
    amt = _first_match(AMOUNT_PATTERNS, text, flags=re.IGNORECASE)
    dat = _first_match(DATE_PATTERNS, text, flags=re.IGNORECASE)
    snd = _first_match(SENDER_PATTERNS, text)

    return {
        "Invoice Number": inv or "Not found",
        "Total Amount": normalize_amount(amt) if amt else "Not found",
        "Issue Date": normalize_date(dat) if dat else "Not found",
        "Sender": snd or "Not found",
    }

# -------------------------
# Main Flow
# -------------------------
if uploaded_files:
    extracted_rows = []
    ocr_debug: List[Tuple[str, List[str]]] = []
    progress = st.progress(0, text="Starting OCR...")
    for idx, uploaded in enumerate(uploaded_files, start=1):
        try:
            progress.progress((idx - 0.5) / len(uploaded_files), text=f"OCR: {uploaded.name}")
            full_text, page_texts = ocr_from_pdf_bytes(uploaded, dpi=dpi, do_threshold=use_threshold, poppler_path=poppler_path)
            ocr_debug.append((uploaded.name, page_texts))
            data = extract_invoice_data(full_text)
            data["Filename"] = uploaded.name
        except Exception as e:
            data = {"Invoice Number": "Error", "Total Amount": "Error", "Issue Date": "Error", "Sender": f"OCR failed: {e}", "Filename": uploaded.name}
        extracted_rows.append(data)
        progress.progress(idx / len(uploaded_files), text=f"Processed: {uploaded.name}")
    progress.empty()

    df = pd.DataFrame(extracted_rows, columns=["Filename", "Invoice Number", "Total Amount", "Issue Date", "Sender"])
    st.success(" Extraction complete!")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        " Download CSV",
        csv_buffer.getvalue(),
        file_name="extracted_invoice_data.csv",
        mime="text/csv",
    )

    # Optional OCR text preview
    if show_ocr_text:
        st.divider()
        st.subheader(" OCR Text Preview (per page)")
        for fname, pages in ocr_debug:
            with st.expander(f"OCR text: {fname}"):
                for i, t in enumerate(pages, start=1):
                    st.markdown(f"**Page {i}**")
                    st.code(t or "(no text)", language="markdown")
else:
    st.info("Upload one or more PDF invoices to begin.")    
