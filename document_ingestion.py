import io
import json
import pdfplumber
import docx
import pptx
import pandas as pd
from PIL import Image
import pytesseract

def ingest_pdf(file_bytes):
    """Extract text from a PDF file."""
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
                else:
                    # Fallback to OCR on image-based pages
                    image = page.to_image(resolution=150)
                    ocr_text = pytesseract.image_to_string(image.original)
                    texts.append(ocr_text)
    except Exception as e:
        texts.append(f"Error reading PDF: {str(e)}")
    return "\n".join(texts)

def ingest_docx(file_bytes):
    """Extract text from a DOCX file."""
    doc = docx.Document(io.BytesIO(file_bytes))
    texts = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(texts)

def ingest_pptx(file_bytes):
    """Extract text from a PPTX file."""
    prs = pptx.Presentation(io.BytesIO(file_bytes))
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text)
    return "\n".join(texts)

def ingest_csv(file_bytes):
    """Extract data from a CSV file; returns a DataFrame and summary text."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df, df.to_csv(index=False)

def ingest_excel(file_bytes):
    """Extract data from an Excel file and return a DataFrame, plus a CSV summary."""
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df, df.to_csv(index=False)

def ingest_json(file_bytes):
    """Extract text from a JSON file by pretty printing."""
    try:
        data = json.load(io.BytesIO(file_bytes))
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error parsing JSON: {str(e)}"

def ingest_txt(file_bytes):
    """Extract text from a TXT file."""
    return file_bytes.decode('utf-8', errors='ignore')

def ingest_image(file_bytes):
    """Run OCR on an image file to extract text."""
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return text
