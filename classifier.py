# Main interface for classifying documents and suggesting relevant fields.

from llm_module import llm_call
from utils.classifier_utils import create_classification_prompt_messages_text, parse_classification

def classify_and_suggest_fields(text: str = "", image_bytes: bytes = None, user_fields: list[str] = None) -> dict:
    """Classifies the document and suggests fields, or uses user-provided fields.

    Args:
        text (optional): The document text (e.g., from OCR).
        image_bytes (optional): Raw image bytes of the document.
        user_fields (optional): A list of fields provided by the user, overriding suggestions.

    Returns:
        A dictionary with "doc_type" (string) and "fields" (list of strings).
    """
    model_identifier = "qwen_25"
    
    # Input validation
    if not text and not image_bytes:
        print("Error: Must provide text or image_bytes for classification.") # Removed
        return {"doc_type": "error", "fields": user_fields or []}

    # Create messages
    messages = create_classification_prompt_messages_text(text)
    if not messages:
        return {"doc_type": "error", "fields": user_fields or []}

    # Call LLM
    print(f"Sending classification prompt with model: {model_identifier}")
    response = llm_call(messages=messages, model_identifier=model_identifier)
    print(f"Received classification response from LLM")#: {response[:200]}...") # Removed

    # Parse response
    if "Error:" in response:
        # print(f"LLM call failed: {response}") # Removed
        return {"doc_type": "error", "fields": user_fields or []}

    parsed_result = parse_classification(response)

    if user_fields:
        return {"doc_type": parsed_result.get("doc_type", "unknown"), "fields": user_fields}
    else:
        return parsed_result

# Note: DEFAULT_FIELDS, parsing logic, and prompt creation in utils/classifier_utils.py

