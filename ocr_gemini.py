# ocr_gemini.py
# Gemini 1.5 Pro Vision OCR for handwritten PDFs
# ------------------------------------------------

import os
import re
import io
import fitz  # PyMuPDF
import google.generativeai as genai

# Load API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise EnvironmentError("ERROR: GOOGLE_API_KEY not set in environment variables.")


def extract_text_with_gemini(pdf_path: str) -> str:
    """
    Extracts handwritten or scanned text from a PDF using Gemini 1.5 Pro Vision.
    Returns a single text string containing all extracted text.

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    -------
    text : str
        OCR-extracted text (raw, unformatted).
    """

    try:
        print(f"[GEMINI OCR] Starting OCR for: {pdf_path}")

        # Load PDF using PyMuPDF (fitz)
        doc = fitz.open(pdf_path)

        final_text = []

        model = genai.GenerativeModel("gemini-2.5-flash")

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=300)

            # Convert pixmap to bytes
            img_bytes = pix.tobytes("png")

            # Gemini OCR Request
            response = model.generate_content(
                [
                    "Extract EXACT text from this handwritten page. Do not summarize. Do not reorganize. Keep math symbols.",
                    {"mime_type": "image/png", "data": img_bytes},
                ]
            )

            page_text = getattr(response, "text", "") or str(response)
            final_text.append(page_text.strip())

            print(f"[GEMINI OCR] Page {page_index+1} extracted ({len(page_text)} chars)")

        output = "\n\n".join(final_text).strip()
        
        # Clean up excessive newlines (max 2)
        output = re.sub(r'\n{3,}', '\n\n', output)
        
        print(f"[GEMINI OCR] Total extracted characters: {len(output)}")

        return output

    except Exception as e:
        print(f"[GEMINI OCR ERROR] {e}")
        return ""
