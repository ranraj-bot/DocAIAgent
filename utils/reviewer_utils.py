# reviewer_utils.py
# Utility functions for the reviewer module.

import json
import re
import base64
import io
from PIL import Image

# --- Helper Function --- 

def _image_bytes_to_base64_url(image_bytes: bytes) -> str | None:
    """Converts image bytes to a base64 data URL.
    (Copied from extractor_utils for modularity in workshop)
    """
    if not image_bytes: return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        format = img.format.lower() if img.format else 'jpeg'
        if format not in ['jpeg', 'png', 'gif', 'webp']:
            format = 'jpeg'
        base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{format};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"Error converting image bytes to base64: {e}")
        return None

# --- Prompt Construction --- 

def build_review_text_prompt(fields_to_review: dict, ocr_text: str = "") -> str:
    """Builds the textual part of the review prompt."""
    extracted_json = json.dumps(fields_to_review)
    fields = list(fields_to_review.keys())
    
    # Example JSON output structure
    example_review = {field: {"status": "PASS or FAIL", "feedback": "..."} for field in fields[:2]} # Show example for first few
    json_format_example = json.dumps(example_review)

    # Construct the core prompt text
    prompt_text = f"""Please act as a meticulous reviewer. Your task is to validate the accuracy of extracted data against the provided document information (primarily the image, secondarily the OCR text).

Extracted Data (JSON Format):
{extracted_json}
"""

    # Append OCR text block if provided
    if ocr_text:
        prompt_text += f"""Reference OCR Text:
Remember words in OCR Text might be jumbled up, and the reading order of neighboring text might not be correct. Keep that in mind and use your judgement to decide if the word order is correct. 
---BEGIN OCR TEXT---
{ocr_text}
---END OCR TEXT---

"""

    # Append instructions and JSON output format
    prompt_text += f"""Instructions:
For each field in the Extracted Data:
1. Compare the extracted value against the primary document source (image if provided, otherwise text) and determine if the extracted value is correct (PASS) or incorrect (FAIL).
2. If the extracted value exists and is not blank, then check if it is present in the document. If present then "PASS". If it is not present then "FAIL".
3. If the extracted value is blank or "" and it is not present in the document, set the status to "PASS".
4. If the status is FAIL, provide brief, specific feedback explaining the error (e.g., "Value not found in image", "Incorrect date format", "Extracted customer name instead of vendor"). If PASS, feedback can be empty or "".

IMPORTANT: Respond ONLY with a single JSON object. The keys of this object should be the exact field names from the Extracted Data. The value for each key should be another JSON object containing two keys: "status" (string: "PASS" or "FAIL") and "feedback" (string).

Donot Give any explanantion in final content. Just JSON response. Example JSON Response Format:
{json_format_example}
"""

    return prompt_text.strip()

def create_review_prompt_messages(fields_to_review: dict, ocr_text: str = "", image_bytes: bytes = None) -> list[dict] | None:
    """Creates the messages payload for the LLM reviewer based on available inputs."""
    if not fields_to_review:
        print("Error: No fields provided for review prompt creation.")
        return None
    if not ocr_text and not image_bytes:
        print("Error: Neither OCR text nor image bytes provided for review.")
        return None

    # 1. Build the core textual prompt instructions
    text_prompt_content = build_review_text_prompt(fields_to_review, ocr_text)
    
    messages = []
    user_content = []

    # 2. Handle image if present
    if image_bytes:
        image_url = _image_bytes_to_base64_url(image_bytes)
        if image_url:
            # Multimodal case: Image first, then text prompt
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            user_content.append({"type": "text", "text": text_prompt_content})
            messages.append({"role": "user", "content": user_content})
            
        else: # Image conversion failed
             if not ocr_text:
                  print("Error: Image conversion failed and no OCR text available for review.")
                  return None
             # Fallback to text-only review (less ideal but possible)
             print("Warning: Image conversion failed. Falling back to text-only review.")
             messages.append({"role": "user", "content": text_prompt_content})
             
    else: # Text-only case (reviewing text against text)
        if not ocr_text: # Should be caught earlier, but double-check
            print("Error: No text provided for text-only review.")
            return None
        messages.append({"role": "user", "content": text_prompt_content})

    return messages

def create_review_prompt_messages_text(fields_to_review: dict, ocr_text: str) -> list[dict] | None:
    """Creates the messages payload for the LLM reviewer using only text."""
    if not fields_to_review:
        print("Error: No fields provided for text-only review prompt creation.")
        return None
    if not ocr_text:
        print("Error: No OCR text provided for text-only review prompt creation.")
        return None

    # Build the core textual instructions 
    text_prompt_content = build_review_text_prompt(fields_to_review, ocr_text)
    
    # Create the simple text-only message format
    messages = [{"role": "user", "content": text_prompt_content}]
    
    return messages

# --- Parsing Function --- 

def parse_review(response: str, fields: list[str]) -> dict:
    """Parses the LLM review response, expecting JSON output."""
    print(f"Parsing review response: {response[:100]}...")
    review_results = {field: {"status": "ERROR", "feedback": "Parsing failed"} for field in fields}
    
    try:
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        json_str = None
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1: 
                json_str = response[first_brace : last_brace + 1].strip()

        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    data_lower_map = {k.lower().strip(): (k, v) for k, v in data.items()}
                    for field in fields:
                        field_lower = field.lower().strip()
                        review_item = None
                        if field_lower in data_lower_map:
                            original_key, review_item = data_lower_map[field_lower]
                        elif field in data:
                            review_item = data[field]
                        
                        if review_item is not None and isinstance(review_item, dict) and 'status' in review_item:
                            status = str(review_item['status']).upper().strip()
                            feedback = str(review_item.get('feedback', '')).strip()
                            if status in ["PASS", "FAIL"]:
                                 review_results[field] = {"status": status, "feedback": feedback}
                            else:
                                 review_results[field] = {"status": "FAIL", "feedback": f"Invalid status: {status}"}
                        elif field in review_results: # Only overwrite if found
                            review_results[field] = {"status": "ERROR", "feedback": "Invalid review item format" if review_item is not None else "Field not found in review response" }
                return review_results
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed for review: {json_err}")
                return review_results # Return initial error dict
        else:
            print("No JSON block found in the review response.")
            return review_results
    except Exception as e:
        print(f"CRITICAL Error during parsing review: {e}")
        return review_results 