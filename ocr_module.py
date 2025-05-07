import time
import os
import pytesseract

from utils.ocr_engines import (
    _call_tesseract_ocr,
    _call_surya_ocr,
    _call_google_vision_ocr,
    _convert_lines_to_reading_order
)

def extract_text(image_bytes: bytes, ocr_engine: str = 'surya') -> str:
    line_data = []
    valid_engines = ['tesseract', 'surya', 'google']

    try:
        selected_engine = ocr_engine.lower()

        if selected_engine == 'tesseract':
            line_data = _call_tesseract_ocr(image_bytes)
        elif selected_engine == 'surya':
            line_data = _call_surya_ocr(image_bytes)
        elif selected_engine in ['google', 'google_vision']:
            line_data = _call_google_vision_ocr(image_bytes)
        else:
            return f"OCR ERROR: Unknown engine '{ocr_engine}'. Choose from: {valid_engines}"

        if not line_data:
             return "OCR INFO: No text detected."

        reading_order_text = _convert_lines_to_reading_order(line_data)
        return reading_order_text

    except pytesseract.TesseractNotFoundError:
         return "OCR ERROR: Tesseract not found."
    except ImportError as ie:
         return f"OCR ERROR: Missing dependency for '{ocr_engine}'. Details: {ie}"
    except RuntimeError as re:
         return f"OCR ERROR: Engine '{ocr_engine}' runtime error. Details: {re}"
    except NameError as ne:
         return f"OCR ERROR: Engine '{ocr_engine}' implementation not found. Details: {ne}"
    except Exception as e:
         return f"OCR ERROR: Unexpected error ({type(e).__name__})."