# document_ingestion.py

import io
import json
import pdfplumber
import docx
import pptx
import pandas as pd
from PIL import Image
import pytesseract

Image.MAX_IMAGE_PIXELS = None

def ingest_pdf(file_bytes):
    """Extract text from a PDF file, with OCR fallback for image-only pages."""
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    texts.append(page_text)
                else:
                    # Fallback to OCR on image-based pages
                    pil_image = page.to_image(resolution=150).original
                    ocr_text = pytesseract.image_to_string(pil_image)
                    texts.append(ocr_text)
    except Exception as e:
        texts.append(f"[PDF Error] {str(e)}")
    return "\n".join(texts)

def ingest_docx(file_bytes):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        texts = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(texts)
    except Exception as e:
        return f"[DOCX Error] {str(e)}"

def ingest_pptx(file_bytes):
    """Extract text from a PPTX file."""
    try:
        prs = pptx.Presentation(io.BytesIO(file_bytes))
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception as e:
        return f"[PPTX Error] {str(e)}"

def ingest_csv(file_bytes):
    """Load CSV into DataFrame and also return CSV text."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        csv_text = df.to_csv(index=False)
        return df, csv_text
    except Exception as e:
        return None, f"[CSV Error] {str(e)}"

def ingest_excel(file_bytes):
    """Load Excel into DataFrame and also return CSV text."""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        csv_text = df.to_csv(index=False)
        return df, csv_text
    except Exception as e:
        return None, f"[Excel Error] {str(e)}"

def ingest_json(file_bytes):
    """Pretty-print JSON content."""
    try:
        data = json.load(io.BytesIO(file_bytes))
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"[JSON Error] {str(e)}"

def ingest_txt(file_bytes):
    """Read text file as UTF-8."""
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"[TXT Error] {str(e)}"

def ingest_image(file_bytes):
    """Run OCR on an image file (PNG/JPG) to extract text."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"[Image OCR Error] {str(e)}"
