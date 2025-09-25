# resume_parser.py
import docx2txt
import PyPDF2
import tempfile
from pdf2image import convert_from_bytes
import pytesseract
import os

# If Windows, set the tesseract path (update if yours is different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(file):
    """
    Extract text from uploaded PDF or DOCX file.
    Handles scanned PDFs using OCR.
    """
    text = ""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        try:
            # Try reading text normally
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # If no text extracted, use OCR
            if not text.strip():
                file.seek(0)  # Reset pointer
                images = convert_from_bytes(file.read())
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"

        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")

    elif filename.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
            file.save(tmp.name)
            text = docx2txt.process(tmp.name)

    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX allowed.")

    return text
