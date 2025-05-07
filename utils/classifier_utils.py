# classifier_utils.py
# Utility functions for the classifier module.

import json
import re
import base64
import io
from PIL import Image

DEFAULT_FIELDS = {
    "invoice": ["Invoice #", "Date", "Total Amount", "Vendor"],
    "bank statement": ["Account Name", "Statement Date", "Closing Balance", "Account Number"],
    "claim form": ["Claim ID", "Patient Name", "Date of Service", "Total Charges"],
    "contract": ["Effective Date", "Party A", "Party B", "Termination Clause"]
}

def _image_bytes_to_base64_url(image_bytes: bytes) -> str | None:
    """Converts image bytes to a base64 data URL."""
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

def build_classification_prompt(text: str = "") -> str:
    """Builds the text prompt for classification and field suggestion."""
    example_json = json.dumps({
        "doc_type": "invoice",
        "fields": ["buyer_address", "buyer_name", "buyer_vat_number", "currency", "invoice_amount", "invoice_date", "invoice_number", "payment_due_date", "po_number", "seller_address", "seller_email", "seller_fax_number", "seller_name", "seller_phone", "seller_vat_number", "seller_website", "shipping_date", "shipto_address", "shipto_name", "subtotal", "total_due_amount", "total_tax"]
    }, indent=2)

    # Adjusted prompt for potentially multimodal input
    prompt = f"""Analyze the provided document.
1. Classify the document type (e.g., Invoice, Bank Statement, Claim Form, Contract, Other).
2. Suggest key fields and table headers relevant for this document type.

"""
    
    prompt += f"""IMPORTANT: Format your response ONLY as a single JSON object with keys "doc_type" (string) and "fields" (list of strings). Do not include any text before or after the JSON object.
Example JSON:
```json
{example_json}
```
If the document type is unclear or doesn't fit common categories, use "other" for the doc_type. Make sure 'fields' is always a list, even if empty.
"""
    return prompt.strip()

def build_classification_prompt_text(text: str = "") -> str:
    """Builds the text prompt for classification and field suggestion."""
    example_json = json.dumps({
        "doc_type": "invoice",
        "fields": ["buyer_address", "buyer_name", "buyer_vat_number", "currency", "invoice_amount", "invoice_date", "invoice_number", "payment_due_date", "po_number", "seller_address", "seller_email", "seller_fax_number", "seller_name", "seller_phone", "seller_vat_number", "seller_website", "shipping_date", "shipto_address", "shipto_name", "subtotal", "total_due_amount", "total_tax"]
    }, indent=2)

    # Adjusted prompt for potentially multimodal input
    prompt = f"""Analyze the provided document.
OCR TEXT
---------
    {text}
---------
1. Classify the document type (e.g., Invoice, Bank Statement, Claim Form, Contract, Other).
2. Suggest key fields and table headers relevant for this document type.

"""
    
    prompt += f"""IMPORTANT: Format your response ONLY as a single JSON object with keys "doc_type" (string) and "fields" (list of strings). Do not include any text before or after the JSON object.
Example JSON:
```json
{example_json}
```
If the document type is unclear or doesn't fit common categories, use "other" for the doc_type. Make sure 'fields' is always a list, even if empty.
"""
    return prompt.strip()

def create_classification_prompt_messages(text: str = "", image_bytes: bytes = None) -> list[dict] | None:
    """Creates the messages payload for the LLM classifier (multimodal capable)."""
    if not text and not image_bytes:
        print("Error: No text or image bytes provided for classification prompt creation.")
        return None

    # Build the core textual instructions (includes OCR text if present)
    text_prompt_content = build_classification_prompt(text)
    
    messages = []
    user_content = []

    if image_bytes:
        image_url = _image_bytes_to_base64_url(image_bytes)
        if image_url:
            # Multimodal case: Image first, then text prompt
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
            # Add a prefix asking to use the image primarily
            user_content.append({"type": "text", "text": text_prompt_content})
            messages.append({"role": "user", "content": user_content})
        else: # Image conversion failed
             if not text: return None # Cannot proceed
             print("Warning: Image conversion failed. Falling back to text-only classification.")
             messages.append({"role": "user", "content": text_prompt_content}) # Use original text prompt
    else: # Text-only case
        if not text: return None # Cannot proceed
        messages.append({"role": "user", "content": text_prompt_content})

    return messages

def create_classification_prompt_messages_text(text: str) -> list[dict] | None:
    """Creates the messages payload for the LLM classifier using only text."""
    if not text:
        print("Error: No text provided for text-only classification prompt creation.")
        return None

    # Build the core textual instructions 
    text_prompt_content = build_classification_prompt_text(text)
    
    # Create the simple text-only message format
    messages = [{"role": "user", "content": text_prompt_content}]
    
    return messages

def parse_classification(response: str) -> dict:
    """Parses the LLM classification response, expecting JSON output.
       Falls back to regex and default fields if JSON parsing fails.
    """
    print(f"Parsing classification response")#: {response[:100]}...")
    doc_type = "error"
    fields = []
    try:
        # Try parsing as JSON first
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        if not json_match:
            json_match = re.search(r"({\s*\"doc_type\"[\s\S]*?})", response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "doc_type" in data and "fields" in data and isinstance(data["fields"], list):
                    doc_type = str(data["doc_type"]).lower().strip() or "unknown"
                    fields = [str(f).strip() for f in data["fields"] if f and isinstance(f, str)]
                    print(f"Parsed classification as JSON: type='{doc_type}', fields={fields}")
                    return {"doc_type": doc_type, "fields": fields}
                else:
                     print("Warning: Parsed JSON missing keys or invalid format.")
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed: {json_err}")
                # Fall through to regex

        # Fallback to regex
        print("Falling back to regex parsing for classification...")
        doc_type_match = re.search(r"Doc(?:ument)?\s*Type:\s*(.*)", response, re.IGNORECASE)
        fields_match = re.search(r"Fields:\s*(\[.*?]|\"?.*\"?(?:,\s*\"?.*\"?)*)", response, re.IGNORECASE | re.DOTALL)

        doc_type = doc_type_match.group(1).strip().lower() if doc_type_match else "unknown"
        fields_str = fields_match.group(1).strip() if fields_match else "[]"

        try:
            if fields_str.startswith("[") and fields_str.endswith("]"):
                 potential_fields = json.loads(fields_str)
                 if isinstance(potential_fields, list):
                     fields = [str(f).strip() for f in potential_fields if f and isinstance(f, str)]
            elif fields_str:
                 fields = [f.strip(' \"\t\r\n') for f in re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', fields_str)]
                 fields = [f for f in fields if f] # Remove empty strings after split
        except (json.JSONDecodeError, Exception) as parse_err:
            print(f"Warning: Could not parse fields string '{fields_str}' via regex: {parse_err}")
            fields = []

        # If regex parsing failed to get fields, use defaults for the detected type
        if not fields and doc_type != "unknown" and doc_type in DEFAULT_FIELDS:
             print(f"Using default fields for doc type: {doc_type}")
             fields = DEFAULT_FIELDS.get(doc_type, []) # Use .get for safety
        elif not fields: # If still no fields (unknown type or no defaults)
             print("Warning: Could not determine fields via JSON or regex, returning empty list.")
             fields = []

        print(f"Parsed classification with regex: type='{doc_type}', fields={fields}")
        return {"doc_type": doc_type, "fields": fields}

    except Exception as e:
        print(f"CRITICAL Error during parsing classification: {e}")
        return {"doc_type": "error", "fields": []} 