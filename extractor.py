# Main interface for extracting key-value pairs from documents.

from llm_module import llm_call
from utils.extractor_utils import create_extraction_prompt_messages, parse_extraction


def extract_key_value_pairs(fields: list[str], ocr_text: str = "", image_bytes: bytes = None) -> dict:
    """Extracts specified key-value pairs using LLM, potentially with multimodal input.

    Args:
        fields: A list of field names (keys) to extract.
        ocr_text (optional): OCR text extracted from the document.
        image_bytes (optional): Raw image bytes of the document.

    Returns:
        A dictionary mapping field names to their extracted values or an error message.
    """
    if not fields:
        print("Warning: No fields specified for extraction.") # Removed
        return {}
    if not ocr_text and not image_bytes:
         print("Error: Must provide ocr_text or image_bytes for extraction.") # Removed
         return {field: "Error: No input" for field in fields}

    # 1. Create prompt messages
    messages = create_extraction_prompt_messages(fields, ocr_text, image_bytes)
    if not messages:
         return {field: "Error: Failed prompt creation" for field in fields}

    # 2. Call LLM
    response = llm_call(messages=messages)

    # 3. Parse response
    if "Error:" in response: 
         print(f"LLM call failed: {response}") # Removed
         return {field: response for field in fields}
         
    extracted_data = parse_extraction(response, fields)
    return extracted_data

# Note: Helper functions and detailed implementation moved to utils/extractor_utils.py