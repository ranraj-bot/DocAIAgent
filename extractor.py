import json
import re
from llm_module import llm_call
# We don't call OCR directly here; text is passed in.
# from ocr_module import extract_text

def parse_extraction(response: str, fields: list[str]) -> dict:
    """Parses the LLM response to extract field values, expecting JSON."""
    print(f"Parsing extraction response: {response[:100]}...")
    # Initialize with None or a placeholder to indicate not found yet
    extracted_data = {field: None for field in fields}
    try:
        # Look for a JSON block ```json ... ``` or just { ... }
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        if not json_match:
            # Try finding a JSON object structure directly
            # Be more robust: Allow optional quotes, handle whitespace
            json_match = re.search(r"({\s*['\"]?.*?['\"]?\s*:[\s\S]*?})", response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
            # Attempt to clean potentially problematic escapes before parsing
            # json_str = json_str.replace('\\"', '"').replace("\'", "'") # Use cautiously
            try:
                data = json.loads(json_str)
                print(f"Parsed extraction as JSON: {data}")
                if isinstance(data, dict):
                    # Populate extracted_data, matching keys (case-insensitive recommended)
                    data_lower = {k.lower().strip(): v for k, v in data.items()}
                    for field in fields:
                        field_lower = field.lower().strip()
                        if field_lower in data_lower:
                            extracted_data[field] = data_lower[field_lower]
                        # Fallback: Check original case if lower fails
                        elif field in data:
                            extracted_data[field] = data[field]
                        else:
                             print(f"Field '{field}' not found in parsed JSON keys.")
                else:
                    print("Warning: Parsed JSON is not a dictionary.")
                return extracted_data # Return data parsed from JSON
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed for extraction string: '{json_str}'. Error: {json_err}")
                # Fall through to return the initialized dict (all None)
                return extracted_data
        else:
            print("No JSON block found in the extraction response.")
            # Optionally, attempt basic regex as a last resort if no JSON
            # print("Falling back to regex parsing for extraction (experimental)...")
            # for field in fields:
            #     # Very basic key: value regex (prone to errors)
            #     pattern = re.compile(rf'['\"]?{re.escape(field)}['\"]?\s*:\s*(['\"]?.*?['\"]?)(?:,\s*|\n|$)′, re.IGNORECASE | re.DOTALL)
            #     match = pattern.search(response)
            #     if match:
            #         value = match.group(1).strip(' \'"')
            #         extracted_data[field] = value
            #         print(f"Regex extracted: {field} = {value}")
            return extracted_data # Return initialized dict if no JSON

    except Exception as e:
        print(f"CRITICAL Error during parsing extraction: {e}\nResponse was: {response}")
        return extracted_data # Return partially filled or empty dict

def extract_fields(text: str, fields: list[str]) -> dict:
    """Extracts specified fields from the text using an LLM."""
    if not text:
        print("Error: No text provided for extraction.")
        return {field: "Error: No text" for field in fields}
    if not fields:
        print("Warning: No fields specified for extraction.")
        return {}

    # Construct the prompt asking for JSON output
    # Ensure fields are quoted correctly if they contain special characters
    fields_json_array = json.dumps(fields)

    prompt = f"""Analyze the following document text and extract the values for the fields specified in the JSON array below.

    Document Text:
    ---BEGIN TEXT---
    {text}
    ---END TEXT---

    Fields to Extract (JSON Array): {fields_json_array}

    IMPORTANT: Respond ONLY with a single JSON object where keys are the exact field names from the array and values are the extracted information from the document text. Do not include any explanatory text, greetings, or markdown formatting before or after the JSON object.
    If a field's value is not found in the text, use `null` or an empty string `""` as the value for that key in the JSON output.

    Example JSON Response Format (If fields were ["Invoice #", "Date", "Missing Field"]):
    {{
      "Invoice #": "INV-12345",
      "Date": "2023-10-27",
      "Missing Field": null
    }}
    """

    print(f"Calling LLM for extraction of fields: {fields}...")
    response = llm_call(prompt)
    print(f"Received extraction response from LLM: {response[:200]}...")

    return parse_extraction(response, fields)

# Simple test (optional)
if __name__ == "__main__":
    # Mock llm_call for standalone testing
    def mock_llm_extract(prompt: str, **kwargs) -> str:
        print(f"--- MOCK LLM Extract Request ---") # Don't print the whole prompt here, can be long
        # Simulate response based on requested fields
        requested_fields_match = re.search(r"Fields to Extract \(JSON Array\): (.*)", prompt)
        requested_fields = []
        if requested_fields_match:
            try:
                requested_fields = json.loads(requested_fields_match.group(1))
            except json.JSONDecodeError:
                print("Mock could not parse requested fields from prompt.")
        
        # Example: Simulate finding some fields and not others
        mock_response = {}
        if "Invoice #" in requested_fields:
            mock_response["Invoice #"] = "MOCK-INV-987"
        if "Date" in requested_fields:
            mock_response["Date"] = "2024-01-15"
        if "Total Amount" in requested_fields:
            mock_response["Total Amount"] = "$555.55"
        if "Vendor" in requested_fields:
            # Simulate missing field
            mock_response["Vendor"] = None 
        # Add any other requested fields as null
        for f in requested_fields:
            if f not in mock_response:
                 mock_response[f] = None

        return json.dumps(mock_response, indent=2)

    # Example mock text (same as classifier)
    mock_text = """MOCK TEXT:
    INVOICE
    Number: INV-000123
    Date: 2025-05-01
    Vendor: ACME Corp
    Total Amount: $1,234.56
    Line Items:
    Item A - $1000.00
    Item B - $234.56"""
    
    fields_to_extract = ["Invoice #", "Date", "Total Amount", "Vendor", "NonExistentField"]

    print("\n--- Testing extract_fields (with mocked LLM) ---")
    from unittest.mock import patch
    with patch('llm_module.llm_call', mock_llm_extract):
        extracted_data = extract_fields(mock_text, fields_to_extract)
        print("\n--- Final Extracted Data ---")
        print(json.dumps(extracted_data, indent=2))

    print("\n--- Testing parse_extraction directly ---")
    test_resp_perfect = '{\n  "Invoice #": "INV-123",\n  "Date": "2023-11-01",\n  "Total Amount": null\n}'
    print(f"Parsing perfect JSON: {parse_extraction(test_resp_perfect, ['Invoice #', 'Date', 'Total Amount'])}")
    test_resp_markdown = '```json\n{\n  "Invoice #": "INV-456",\n  "Date": "2023-11-02"\n}\n```'
    print(f"Parsing markdown JSON: {parse_extraction(test_resp_markdown, ['Invoice #', 'Date'])}")
    test_resp_text = 'Some text before {\n  "Invoice #": "INV-789"\n} and after.'
    print(f"Parsing JSON with surrounding text: {parse_extraction(test_resp_text, ['Invoice #'])}")
    test_resp_fail = 'This response has no json.'
    print(f"Parsing failing response: {parse_extraction(test_resp_fail, ['Invoice #'])}") 