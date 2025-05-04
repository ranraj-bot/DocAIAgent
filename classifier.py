import re
import json
from llm_module import llm_call

DEFAULT_FIELDS = {
    "invoice": ["Invoice #", "Date", "Total Amount", "Vendor"],
    "bank statement": ["Account Name", "Statement Date", "Closing Balance", "Account Number"],
    "claim form": ["Claim ID", "Patient Name", "Date of Service", "Total Charges"],
    "contract": ["Effective Date", "Party A", "Party B", "Termination Clause"]
}

def parse_classification(response: str) -> dict:
    """Parses the LLM response to extract doc type and fields."""
    print(f"Parsing classification response: {response[:100]}...")
    try:
        # Try parsing as JSON first (more robust)
        # Look for a JSON block ```json ... ``` or just {"doc_type": ...}
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        if not json_match:
            # Try finding a JSON object structure directly
            json_match = re.search(r"(\{\s*\"doc_type\"[\s\S]*?\})", response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # Attempt to clean common escape issues, but be cautious
                # json_str = json_str.replace('\\"', '"').replace('\\n', '\n') # Usually not needed if LLM follows prompt
                data = json.loads(json_str)
                print(f"Parsed classification as JSON: {data}")
                if "doc_type" in data and "fields" in data and isinstance(data["fields"], list):
                    doc_type = str(data["doc_type"]).lower().strip()
                    fields = [str(f).strip() for f in data["fields"] if isinstance(f, str)]
                    return {"doc_type": doc_type, "fields": fields}
                else:
                    print("Warning: Parsed JSON missing 'doc_type'/'fields' or 'fields' is not a list.")
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed for string: '{json_str}'. Error: {json_err}")
                # Fall through to regex parsing

        # Fallback to simple regex parsing if JSON fails or wasn't found
        print("Falling back to regex parsing for classification...")
        doc_type_match = re.search(r"Doc(?:ument)?\s*Type:\s*(.*)", response, re.IGNORECASE)
        # Match fields either in a list format or as comma-separated strings
        fields_match = re.search(r"Fields:\s*(\[.*?\]|\"?.*\"?(?:,\s*\"?.*\"?)*)", response, re.IGNORECASE | re.DOTALL)

        doc_type = doc_type_match.group(1).strip().lower() if doc_type_match else "unknown"
        fields_str = fields_match.group(1).strip() if fields_match else "[]"

        fields = []
        try:
            # Handle JSON-like list format ["a", "b"]
            if fields_str.startswith("[") and fields_str.endswith("]"):
                 # Prefer json.loads over eval for safety
                 potential_fields = json.loads(fields_str)
                 if isinstance(potential_fields, list):
                     fields = potential_fields
            # Handle comma-separated strings, potentially quoted
            elif fields_str:
                 # Regex to handle commas inside quotes: splits by comma unless inside quotes
                 fields = [f.strip(' "\t\r\n') for f in re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', fields_str)]

            # Final cleaning of the list
            fields = [str(f).strip() for f in fields if isinstance(f, (str, int, float)) and str(f).strip()] # Allow numbers too?

        except (json.JSONDecodeError, Exception) as parse_err:
            print(f"Warning: Could not parse fields string '{fields_str}': {parse_err}")
            fields = [] # Ensure fields is an empty list on error

        # If parsing failed to get fields, use defaults for the detected type
        if not fields and doc_type != "unknown" and doc_type in DEFAULT_FIELDS:
             print(f"Using default fields for doc type: {doc_type}")
             fields = DEFAULT_FIELDS[doc_type]

        print(f"Parsed classification with regex: type='{doc_type}', fields={fields}")
        return {"doc_type": doc_type, "fields": fields}

    except Exception as e:
        print(f"CRITICAL Error during parsing classification: {e}\nResponse was: {response}")
        return {"doc_type": "error", "fields": []}


def classify_and_suggest_fields(text: str, user_fields: list[str] = None) -> dict:
    """Classifies the document based on text and suggests fields, or uses user-provided fields."""
    if user_fields:
        # Even if user provides fields, classify for context but prioritize user's list.
        prompt = f"""Analyze the following document text only to classify its type.
        Ignore any previous instructions about suggesting fields for this specific step.
        What is the document type (e.g., Invoice, Bank Statement, Claim Form, Contract, Other)?

        Respond ONLY with the document type classification.

        Document Text:
        ---BEGIN TEXT---
        {text}
        ---END TEXT---
        """
        print("Calling LLM for classification only (user fields provided)...")
        # A simplified call just to get the type if needed, response parsing might differ
        # resp = llm_call(prompt)
        # For now, let's skip the LLM call here if fields are provided to save tokens/time
        # doc_type = "unknown (user fields provided)" # Placeholder
        # Or maybe run the full classification but ignore its field suggestions:
        full_classification_result = classify_and_suggest_fields(text) # Call recursively without user_fields
        doc_type = full_classification_result.get("doc_type", "unknown")
        print(f"Classification ran, detected type: {doc_type}, but using user-provided fields.")
        return {"doc_type": doc_type, "fields": user_fields}
    else:
        # Ask LLM to classify AND suggest fields, requesting JSON output
        prompt = f"""Analyze the following document text.
        1. Classify the document type (e.g., Invoice, Bank Statement, Claim Form, Contract, Other).
        2. Suggest key fields and table headers relevant for this document type.

        IMPORTANT: Format your response ONLY as a single JSON object with keys "doc_type" (string) and "fields" (list of strings). Do not include any text before or after the JSON object.
        Example JSON:
        ```json
        {{
          "doc_type": "invoice",
          "fields": ["Invoice Number", "Invoice Date", "Vendor Name", "Total Amount"]
        }}
        ```
        If the document type is unclear or doesn't fit common categories, use "other" for the doc_type. Make sure 'fields' is always a list, even if empty.

        Document Text:
        ---BEGIN TEXT---
        {text}
        ---END TEXT---
        """
        print("Calling LLM for classification and field suggestion... Awaiting response...")
        resp = llm_call(prompt)
        print(f"Received classification response from LLM: {resp[:200]}...")
        return parse_classification(resp)

# Simple test (optional)
if __name__ == "__main__":
    # Mock llm_call for standalone testing without API keys
    def mock_llm_func(prompt: str, **kwargs) -> str:
        print(f"--- MOCK LLM Request ---\n{prompt}\n--- End MOCK ---")
        if "ONLY as a single JSON object" in prompt:
             # Simulate JSON response for classification + fields
             return """
{
  "doc_type": "invoice",
  "fields": ["Number", "Date", "Vendor", "Total Amount", "Line Items"]
}
"""
        else:
             # Simulate simple text response for classification only
             return "Invoice"

    # Example mock text
    mock_text = """MOCK TEXT:
    INVOICE
    Number: INV-000123
    Date: 2025-05-01
    Vendor: ACME Corp
    Total Amount: $1,234.56
    Line Items:
    Item A - $1000.00
    Item B - $234.56"""

    print("\n--- Testing parser directly ---")
    test_resp_json = '```json\n{\n  "doc_type": "invoice",\n  "fields": ["Invoice Number", "Invoice Date", "Vendor Name", "Total Amount"]\n}\n```'
    print(f"Parsing JSON: {parse_classification(test_resp_json)}")
    test_resp_json_direct = '{\n  "doc_type": "bank statement",\n  "fields": ["Acc Name", "Stmt Date", "Balance"]\n}'
    print(f"Parsing Direct JSON: {parse_classification(test_resp_json_direct)}")
    test_resp_regex = 'Doc Type: Contract\nFields: ["Party A", "Party B", "Effective Date"]'
    print(f"Parsing Regex: {parse_classification(test_resp_regex)}")
    test_resp_malformed = '{"doc_type": "claim", "fields": "Claim ID, Patient"}' # Malformed fields
    print(f"Parsing Malformed (expect fallback/defaults): {parse_classification(test_resp_malformed)}")
    test_resp_nofields_key = '{"doc_type": "other"}'
    print(f"Parsing Missing Fields Key: {parse_classification(test_resp_nofields_key)}")


    print("\n--- Testing classify_and_suggest_fields (with mocked LLM) ---")
    from unittest.mock import patch

    with patch('llm_module.llm_call', mock_llm_func):
        print("\nTesting classification with mock text:")
        result = classify_and_suggest_fields(mock_text)
        print(f"Result: {result}")

        print("\nTesting with user-provided fields:")
        user_flds = ["Invoice #", "Total Amount"]
        # Note: The updated logic might call the function recursively now
        result_user = classify_and_suggest_fields(mock_text, user_fields=user_flds)
        print(f"Result (user fields): {result_user}")
