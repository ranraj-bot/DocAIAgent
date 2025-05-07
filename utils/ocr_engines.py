# ocr_engines.py
# Contains the implementation details for different OCR engines and reading order logic.

from PIL import Image
import io
import pandas as pd
import pytesseract
import os
import time
import statistics

# --- Lazy Loading & Imports for Specific Engines ---

# Surya
surya_recognition_predictor = None
surya_detection_predictor = None
surya_imported = False
torch_imported = False
try:
    import torch
    torch_imported = True
except ImportError:
    print("WARNING: PyTorch not found. Surya OCR will not be available.")

# Google Vision
google_vision_client = None
google_vision_imported = False
vision = None # Define vision module variable
try:
    from google.cloud import vision
    google_vision_imported = True
except ImportError:
    print("WARNING: google-cloud-vision library not found. Google Vision OCR will not be available.")
except Exception as e:
    print(f"WARNING: Error during initial Google Vision import/setup: {e}")
    google_vision_imported = False


def _maybe_init_surya():
    """Initializes Surya predictors if not already done."""
    global surya_recognition_predictor, surya_detection_predictor, surya_imported
    if not torch_imported:
         raise ImportError("PyTorch is required for Surya OCR but not installed.")

    if surya_recognition_predictor is None or surya_detection_predictor is None:
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            print("Initializing Surya OCR models (this may take a while)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device} for Surya")
            start_init = time.time()
            surya_detection_predictor = DetectionPredictor(device=device)
            surya_recognition_predictor = RecognitionPredictor(device=device)
            end_init = time.time()
            surya_imported = True
            print(f"Surya OCR models initialized (took {end_init - start_init:.2f}s).")
        except ImportError as e:
            print(f"ERROR: Failed to import Surya OCR components: {e}. Please install surya-ocr and its dependencies (torch, torchvision, timm).")
            surya_imported = False
            raise
        except Exception as e:
             print(f"ERROR: Failed to initialize Surya OCR models: {e}")
             surya_imported = False
             raise

def _maybe_init_google_vision():
    """Initializes Google Vision client if not already done."""
    global google_vision_client
    if google_vision_imported and google_vision_client is None:
         try:
             print("Initializing Google Vision Client...")
             google_vision_client = vision.ImageAnnotatorClient()
             print("Google Vision Client Initialized.")
         except Exception as e:
              print(f"ERROR: Failed to initialize Google Vision Client: {e}")
              # Optionally re-raise or handle specific auth errors here
              raise RuntimeError(f"Failed to initialize Google Vision client: {e}")


# --- Standardized Line Output Format --- 
# { 'text': str, 'bbox': [xmin, ymin, xmax, ymax], 'confidence': float (0-100 scale) }

# --- Helper Function for Google Vision --- 
def _vertices_to_bbox(vertices) -> list[int]:
    """Converts a list of Vision API vertices [{x,y},...] to [xmin, ymin, xmax, ymax]."""
    x_coords = [v.x for v in vertices if hasattr(v, 'x') and v.x is not None]
    y_coords = [v.y for v in vertices if hasattr(v, 'y') and v.y is not None]
    if not x_coords or not y_coords: return [0, 0, 0, 0]
    try:
      xmin = int(min(x_coords)); ymin = int(min(y_coords))
      xmax = int(max(x_coords)); ymax = int(max(y_coords))
      if xmin >= xmax or ymin >= ymax: return [0,0,0,0]
      return [xmin, ymin, xmax, ymax]
    except ValueError: return [0,0,0,0]

# --- Google Vision OCR Implementation --- 
def _call_google_vision_ocr(image_bytes: bytes, language_hint: str = None) -> list[dict]:
    """Implementation for calling Google Vision OCR with retry logic."""
    if not google_vision_imported: raise ImportError("Google Vision library not available.")
    _maybe_init_google_vision() # Ensure client is ready (raises RuntimeError on failure)
    if google_vision_client is None: raise RuntimeError("Google Vision client could not be initialized.")

    max_retries = 3
    base_delay = 1.0 # seconds

    for attempt in range(max_retries):
        try:
            print(f"Calling Google Cloud Vision OCR (Attempt {attempt + 1}/{max_retries})...")
            image = vision.Image(content=image_bytes)
            text_params = vision.TextDetectionParams(enable_text_detection_confidence_score=True)
            image_context = vision.ImageContext(text_detection_params=text_params, language_hints=[language_hint] if language_hint else [])
            
            # *** API Call ***
            response = google_vision_client.text_detection(image=image, image_context=image_context)

            # Check for API level errors returned in the response object
            if response.error.message:
                # Treat this as a potentially transient error for retry purposes
                raise Exception(f"Google Vision API Error: {response.error.message}")

            # --- Process successful response --- 
            if not response.full_text_annotation or not response.full_text_annotation.pages:
                print("Google Vision returned no text annotation or pages.")
                return [] # Success, but no data

            line_data = []
            page = response.full_text_annotation.pages[0]
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    word_confidences = []
                    word_texts = []
                    if paragraph.words:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            word_texts.append(word_text)
                            if word.confidence > 0: word_confidences.append(word.confidence)
                    para_text = " ".join(word_texts).strip()
                    if not para_text: continue
                    para_bbox = _vertices_to_bbox(paragraph.bounding_box.vertices)
                    avg_confidence = (sum(word_confidences) / len(word_confidences) * 100) if word_confidences else 0.0
                    if para_bbox == [0, 0, 0, 0]: continue
                    line_data.append({'text': para_text, 'bbox': para_bbox, 'confidence': avg_confidence})
            
            print(f"Google Vision OCR generated {len(line_data)} lines (paragraphs).")
            return line_data # *** Success: return results ***

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {type(e).__name__} - {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Failing Google Vision call.")
                # Option 1: Return empty list (consistent with other errors here)
                return [] 
                # Option 2: Re-raise the last exception
                # raise e 
            else:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

    # Should only be reached if all retries fail and we chose not to re-raise
    print("Returning empty list after all Google Vision retries failed.")
    return []

# --- Tesseract OCR Implementation --- 
def _call_tesseract_ocr(image_bytes: bytes) -> list[dict]:
    """Implementation for calling Tesseract OCR."""
    print("Calling Tesseract OCR...")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        ocr_df = pytesseract.image_to_data(img, config='--psm 6', output_type=pytesseract.Output.DATAFRAME, lang='eng')
        ocr_df = ocr_df.dropna(subset=['text', 'conf', 'left', 'top', 'width', 'height', 'line_num'])
        ocr_df = ocr_df[ocr_df.conf != -1]
        for col in ['left', 'top', 'width', 'height', 'conf', 'line_num']:
             ocr_df[col] = pd.to_numeric(ocr_df[col], errors='coerce')
        ocr_df = ocr_df.dropna()
        ocr_df['text'] = ocr_df['text'].astype(str).str.strip()
        ocr_df = ocr_df[ocr_df.text != '']
        ocr_df = ocr_df[ocr_df.conf > 30]
        if ocr_df.empty: return []
        grouped_lines = ocr_df.groupby('line_num')
        line_data = []
        for _, words_in_line in grouped_lines:
            if words_in_line.empty: continue
            words_in_line = words_in_line.sort_values(by='left')
            line_text = " ".join(words_in_line['text'])
            xmin = words_in_line['left'].min(); ymin = words_in_line['top'].min()
            xmax = (words_in_line['left'] + words_in_line['width']).max()
            ymax = (words_in_line['top'] + words_in_line['height']).max()
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            avg_confidence = words_in_line['conf'].mean()
            if line_text and avg_confidence > 0:
                 line_data.append({'text': line_text, 'bbox': bbox, 'confidence': avg_confidence})
        print(f"Tesseract OCR generated {len(line_data)} lines from words (word conf > 30).")
        return line_data
    except pytesseract.TesseractNotFoundError: print("ERROR: Tesseract executable not found."); raise
    except Exception as e: print(f"Error during Tesseract line processing: {e}"); return []

# --- Surya OCR Implementation --- 
def _call_surya_ocr(image_bytes: bytes) -> list[dict]:
    """Implementation for calling Surya OCR."""
    _maybe_init_surya()
    if not surya_imported: raise RuntimeError("Surya OCR components failed to initialize or import.")
    print("Calling Surya OCR...")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        langs = None
        print("Running Surya detection and recognition...")
        predictions = surya_recognition_predictor([img], [langs], surya_detection_predictor)
        if not predictions or not predictions[0].text_lines: return []
        line_data = []
        min_surya_confidence = 0.5
        for line in predictions[0].text_lines:
            if line.confidence < min_surya_confidence: continue
            try: bbox = [int(coord) for coord in line.bbox]; assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
            except: continue
            line_text = line.text.strip() if line.text else ""
            if not line_text: continue
            line_data.append({'text': line_text, 'bbox': bbox, 'confidence': line.confidence * 100})
        print(f"Surya OCR generated {len(line_data)} lines (line conf > {min_surya_confidence}).")
        return line_data
    except Exception as e: print(f"Error during Surya OCR processing: {e}"); return []

# --- Reading Order Conversion Implementation --- 
def _convert_lines_to_reading_order(line_data: list[dict]) -> str:
    """Implementation for converting line data to reading order string."""
    if not line_data: return ""
    heights = []
    for line in line_data:
        try:
            bbox = line['bbox']
            if len(bbox) == 4: height = bbox[3] - bbox[1]; 
            if height > 0: heights.append(height)
        except: continue
    if not heights:
        print("Warning: Using simple sort for reading order (no heights).")
        line_data.sort(key=lambda line: (line.get('bbox', [0,0,0,0])[1], line.get('bbox', [0,0,0,0])[0]))
        return "\n".join([line.get('text', '') for line in line_data])
    try: median_height = statistics.median(heights)
    except: median_height = 10
    y_tolerance = median_height * 0.7
    print(f"Median line height: {median_height:.2f}, Y-tolerance: {y_tolerance:.2f}")
    line_data.sort(key=lambda line: line.get('bbox', [0,0,0,0])[1])
    lines = []
    current_line_boxes = []
    current_line_ref_y = -1
    for box in line_data:
        try:
            box_top = box['bbox'][1]
            if not current_line_boxes or abs(box_top - current_line_ref_y) > y_tolerance:
                if current_line_boxes: current_line_boxes.sort(key=lambda b: b['bbox'][0]); lines.append(" ".join([b.get('text', '') for b in current_line_boxes]))
                current_line_boxes = [box]; current_line_ref_y = box_top
            else:
                current_line_boxes.append(box); current_line_ref_y = min(current_line_ref_y, box_top)
        except: continue
    if current_line_boxes: current_line_boxes.sort(key=lambda b: b['bbox'][0]); lines.append(" ".join([b.get('text', '') for b in current_line_boxes]))
    return "\n".join(lines) 