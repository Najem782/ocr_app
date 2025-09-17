import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
import re
import pandas as pd
import tempfile
from io import BytesIO
import concurrent.futures

st.title("Invoice Extractor (OCR from PDF)")
st.write("Upload one or more scanned PDF invoices. The app will extract invoice number, total amount, date, and sender using OCR.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

def ocr_from_pdf_bytes(pdf_bytes):
    # Use a lower DPI to speed up processing (150 DPI should be sufficient for OCR)
    images = convert_from_bytes(pdf_bytes.read(), dpi=150)
    full_text = ""
    for page_number, page in enumerate(images, start=1):
        text = pytesseract.image_to_string(page, lang='eng', config="--psm 6 --oem 3")
        full_text += text + "\n\n"
    return full_text

def extract_invoice_data(text):
    invoice_number_pattern = r"Delivery Advice No\.\s*[:\-]?\s*(\d{8,})"
    total_amount_pattern = r"Invoice\s+USD\s+(\d+(?:,\d{3})*(?:\.\d{2}))"
    issue_date_pattern = r"Dated\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})"
    sender_pattern = r"(?i)([A-Za-z ]+Emerging Markets FZE)"

    invoice_number = re.search(invoice_number_pattern, text)
    total_amount = re.search(total_amount_pattern, text)
    issue_date = re.search(issue_date_pattern, text)
    sender = re.search(sender_pattern, text)

    return {
        "Invoice Number": invoice_number.group(1) if invoice_number else "Not found",
        "Total Amount (USD)": total_amount.group(1) if total_amount else "Not found",
        "Issue Date": issue_date.group(1) if issue_date else "Not found",
        "Sender": sender.group(1).strip() if sender else "Not found"
    }

def process_file(uploaded_file):
    text = ocr_from_pdf_bytes(uploaded_file)
    data = extract_invoice_data(text)
    data['Filename'] = uploaded_file.name
    return data

if uploaded_files:
    extracted_data = []

    with st.spinner("Extracting data from uploaded PDFs..."):
        # Use ThreadPoolExecutor to process files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_file, uploaded_files)
            extracted_data.extend(results)

    df = pd.DataFrame(extracted_data)
    st.success("Extraction complete!")
    st.dataframe(df)

    # Generate a dynamic filename based on the first invoice number (or fallback if not found)
    first_invoice_number = extracted_data[0]["Invoice Number"] if extracted_data else "invoice_data"
    file_name = f"{first_invoice_number}_invoice_data.csv" if first_invoice_number != "Not found" else "invoice_data.csv"

    # Create CSV in memory
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Move to the start of the buffer for download

    # Provide the download button with the dynamic filename
    st.download_button("Download CSV", csv_buffer.getvalue(), file_name=file_name, mime="text/csv")
