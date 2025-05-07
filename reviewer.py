# Main interface for reviewing extracted document fields.

from llm_module import llm_call
from utils.reviewer_utils import create_review_prompt_messages, parse_review


def review_fields(fields_to_review: dict, ocr_text: str = "", image_bytes: bytes = None) -> dict:
    """Reviews extracted fields against document info (image and/or text) using an LLM.

    Args:
        fields_to_review: Dict where keys are field names and values are the extracted values.
        ocr_text (optional): OCR text extracted from the document for reference.
        image_bytes (optional): Raw image bytes of the document (primary source if provided).

    Returns:
        A dictionary mapping field names to review results (status: PASS/FAIL, feedback: str).
    """
    if not fields_to_review:
        print("Warning: No extracted data provided for review.")
        return {}
    # Basic check: Ensure at least one source (image or text) is available
    if not ocr_text and not image_bytes:
         print("Error: Must provide ocr_text or image_bytes for review.")
         return {field: {"status": "ERROR", "feedback": "No source document info"} for field in fields_to_review}

    # 1. Create prompt messages using the utility function
    messages = create_review_prompt_messages(fields_to_review, ocr_text, image_bytes)
    if not messages:
        # Error logged within create_review_prompt_messages
        return {field: {"status": "ERROR", "feedback": "Failed prompt creation"} for field in fields_to_review}

    # 2. Call the LLM (using the "reviewer" identifier)
    model_identifier = "qwen_25_vl"
    print(f"Calling LLM for review (using '{model_identifier}')...")
    response = llm_call(messages=messages, model_identifier=model_identifier)
    print(f"Received review response from LLM: {response[:200]}...")

    # 3. Parse the response using the utility function
    if "Error:" in response: # Handle errors returned directly from llm_call
         print(f"LLM call failed: {response}")
         return {field: {"status": "ERROR", "feedback": response} for field in fields_to_review}

    review_results = parse_review(response, list(fields_to_review.keys()))
    return review_results

# Note: Implementation details (prompting, parsing, helpers) in utils/reviewer_utils.py
