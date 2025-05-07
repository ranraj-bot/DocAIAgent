# extractor_utils.py
# Utility functions for the extractor module.

import json
import re
import base64
import io
from PIL import Image

# --- Helper Functions --- 

def _image_bytes_to_base64_url(image_bytes: bytes) -> str | None:
    """Converts image bytes to a base64 data URL.
    Attempts to determine the image type, defaults to jpeg.
    Returns None if conversion fails.
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

def build_extraction_text_prompt(fields: list[str], ocr_text: str = "") -> str:
    """Builds the textual part of the extraction prompt."""
    field_instructions = "\n".join(f"{i+1}. {field}" for i, field in enumerate(fields))
    json_dict = {field: "..." for field in fields}
    json_format_example = json.dumps(json_dict)

    prompt_text = f"""Follow the below instructions and extract field(s) from the provided document. If value is not present for a field then "" should be provided. If there are more than 1 value for a field, give all the values as an array.

Extract the following fields:
{field_instructions}

"""
    if ocr_text:
        prompt_text += f"""Following is the OCR text extracted from the document. It may contain missing text, incorrect layout, or OCR errors. Use it as a reference alongside any provided image:
---BEGIN OCR TEXT---
{ocr_text}
---END OCR TEXT---

"""
    prompt_text += f"""The output should be formatted ONLY as a single flattened JSON object. Do not give any additional explanation.
OUTPUT JSON FORMAT:
{json_format_example}
"""
    return prompt_text.strip()

def create_extraction_prompt_messages(fields: list[str], ocr_text: str = "", image_bytes: bytes = None) -> list[dict] | None:
    """Creates the messages payload for the LLM based on available inputs."""
    if not fields: return None
    if not ocr_text and not image_bytes: return None

    text_prompt_content = build_extraction_text_prompt(fields, ocr_text)
    messages = []
    user_content = []

    if image_bytes:
        image_url = _image_bytes_to_base64_url(image_bytes)
        if image_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
            user_content.append({"type": "text", "text": text_prompt_content})
            messages.append({"role": "user", "content": user_content})
        else: 
             if not ocr_text: return None
             print("Warning: Image conversion failed. Falling back to text-only extraction.")
             messages.append({"role": "user", "content": text_prompt_content})
    else:
        messages.append({"role": "user", "content": text_prompt_content})

    return messages

def create_extraction_prompt_messages_text(fields: list[str], ocr_text: str) -> list[dict] | None:
    """Creates the messages payload for the LLM extractor using only text."""
    if not fields:
        print("Error: No fields provided for text-only extraction prompt creation.")
        return None
    if not ocr_text:
        print("Error: No OCR text provided for text-only extraction prompt creation.")
        return None

    # Build the core textual instructions (requires OCR text for this function)
    text_prompt_content = build_extraction_text_prompt(fields, ocr_text)
    
    # Create the simple text-only message format
    messages = [{"role": "user", "content": text_prompt_content}]
    
    return messages

def parse_extraction(response: str, fields: list[str]) -> dict:
    """Parses the LLM response to extract field values, expecting JSON."""
    #print(f"Parsing extraction response: {response[:100]}...")
    extracted_data = {field: None for field in fields}
    try:
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        if not json_match:
            json_match = re.search(r"({\s*['\"]?.*?['\"]?\s*:[\s\S]*?})", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    data_lower = {k.lower().strip(): v for k, v in data.items()}
                    for field in fields:
                        field_lower = field.lower().strip()
                        if field_lower in data_lower:
                            extracted_data[field] = data_lower[field_lower]
                        elif field in data:
                            extracted_data[field] = data[field]
                return extracted_data
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed: {json_err}")
                return extracted_data
        else:
            return extracted_data
    except Exception as e:
        print(f"CRITICAL Error during parsing extraction: {e}")
        return extracted_data 