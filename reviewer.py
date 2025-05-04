import json
import re
from llm_module import llm_call

def parse_review(response: str, fields: list[str]) -> dict:
    """Parses the LLM response to extract review status (PASS/FAIL) and feedback for each field."""
    print(f"Parsing review response: {response[:100]}...")
    # Initialize review results
    review_results = {field: {"status": "ERROR", "feedback": "Parsing failed"} for field in fields}
    
    try:
        # Expecting JSON output for review results
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        json_str = None
        if json_match:
            json_str = json_match.group(1).strip()
            print("Found JSON block using ```json markers.")
        else:
            # Fallback: Find the first '{' and the last '}'
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = response[first_brace : last_brace + 1].strip()
                print("Found JSON block using first/last brace heuristic.")
            else:
                print("Could not find a JSON block using markers or braces.")

        if json_str:
            try:
                # Clean potential unicode escapes that json might barf on
                # json_str = json_str.encode('utf-8').decode('unicode_escape') # Be careful with this
                data = json.loads(json_str)
                print(f"Parsed review as JSON: {data}")
                if isinstance(data, dict):
                    # Process the parsed dictionary
                    # Create a mapping from lower-case field name to original field name and its value
                    data_lower_map = {k.lower().strip(): (k, v) for k, v in data.items()}

                    for field in fields:
                        field_lower = field.lower().strip()
                        review_item = None # Initialize review_item

                        # Try case-insensitive match first using the map
                        if field_lower in data_lower_map:
                            original_key, review_item = data_lower_map[field_lower]
                            print(f"Found review item for '{field}' (matched lower key '{field_lower}' to original '{original_key}')")
                        # Fallback: Try exact original case match if lowercase failed (should be rare if map is comprehensive)
                        elif field in data:
                            review_item = data[field]
                            print(f"Found review item for '{field}' (exact case match - fallback)")

                        if review_item is not None:
                            # Check if the retrieved item has the expected structure
                            if isinstance(review_item, dict) and 'status' in review_item:
                                status = str(review_item['status']).upper().strip()
                                feedback = str(review_item.get('feedback', '')).strip()
                                # Allow specific valid statuses
                                if status in ["PASS", "FAIL", "NEEDS_IMPROVEMENT"]: # NEEDS_IMPROVEMENT might be useful later
                                     review_results[field] = {"status": status, "feedback": feedback}
                                else:
                                     print(f"Warning: Invalid status '{status}' for field '{field}'. Defaulting to FAIL.")
                                     review_results[field] = {"status": "FAIL", "feedback": f"Invalid status received: {status}"}
                            else:
                                print(f"Warning: Invalid format for field '{field}' review item: {review_item}. Expected dict with 'status'. Defaulting to ERROR.")
                                review_results[field] = {"status": "ERROR", "feedback": f"Invalid review item format: {review_item}"}
                        else:
                             # Only log if the field wasn't found by either method
                             print(f"Warning: Field '{field}' not found in review response keys (checked case-insensitive and exact).")
                             review_results[field] = {"status": "ERROR", "feedback": "Field not found in review response"}
                else:
                     print("Warning: Parsed review JSON is not a dictionary.")
                     # Keep initial error state for all fields
                return review_results
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parsing failed for review string: '{json_str}'. Error: {json_err}")
                # Keep initial error state
                return review_results
        else:
             print("No JSON block found in the review response.")
             # Keep initial error state
             return review_results

    except Exception as e:
        print(f"CRITICAL Error during parsing review: {e}\nResponse was: {response}")
        return review_results # Return initialized with errors

def review_fields(text: str, extracted_data: dict) -> dict:
    """Uses an LLM to review the extracted fields against the original text."""
    if not text:
        print("Error: No text provided for review.")
        return {field: {"status": "ERROR", "feedback": "No text provided"} for field in extracted_data}
    if not extracted_data:
        print("Warning: No extracted data provided for review.")
        return {}

    extracted_json = json.dumps(extracted_data, indent=2)
    fields = list(extracted_data.keys())

    prompt = f"""Please act as a meticulous reviewer. Your task is to validate the accuracy of extracted data against the provided document text.

    Document Text:
    ---BEGIN TEXT---
    {text}
    ---END TEXT---

    Extracted Data (JSON Format):
    ```json
    {extracted_json}
    ```

    Instructions:
    For each field in the Extracted Data:
    1. Compare the extracted value against the Document Text.
    2. Determine if the extracted value is correct (PASS) or incorrect (FAIL).
    3. If the status is FAIL, provide a brief, specific feedback explaining the error (e.g., "Picked up customer number instead", "Date format is wrong", "Value not found in text"). If PASS, feedback can be empty or "-".

    IMPORTANT: Respond ONLY with a single JSON object. The keys of this object should be the exact field names from the Extracted Data. The value for each key should be another JSON object containing two keys: "status" (string: "PASS" or "FAIL") and "feedback" (string).

    Example JSON Response Format:
    {{
      "Field Name 1": {{ "status": "PASS", "feedback": "-" }},
      "Field Name 2": {{ "status": "FAIL", "feedback": "Extracted value is from a different section." }},
      "Field Name 3": {{ "status": "PASS", "feedback": "" }}
    }}
    """

    print(f"Calling LLM for review of fields: {fields}...")
    response = llm_call(prompt)
    print(f"Received review response from LLM: {response[:200]}...")

    return parse_review(response, fields)

# Simple test (optional)
if __name__ == "__main__":
    # Mock llm_call for standalone testing
    def mock_llm_review(prompt: str, **kwargs) -> str:
        print(f"--- MOCK LLM Review Request ---")
        # Simulate response based on provided extracted data in prompt
        extracted_match = re.search(r"Extracted Data \(JSON Format\):\n\s*```json\n(.*?)\n```", prompt, re.DOTALL)
        extracted_data_for_review = {}
        if extracted_match:
            try:
                extracted_data_for_review = json.loads(extracted_match.group(1))
            except json.JSONDecodeError:
                print("Mock could not parse extracted data from prompt.")
        
        # Example: Simulate PASS/FAIL based on field names
        mock_review = {}
        for field, value in extracted_data_for_review.items():
            if "fail" in field.lower() or value is None:
                mock_review[field] = {"status": "FAIL", "feedback": f"Mock feedback: Issue detected with {field}"}
            else:
                mock_review[field] = {"status": "PASS", "feedback": "-"}
        
        # Ensure the output is valid JSON
        return json.dumps(mock_review, indent=2)

    # Example mock text and extracted data
    mock_text = "INVOICE\nNumber: Real-123\nDate: 2025-01-01\nTotal: $100.00"
    mock_extracted = {
        "Invoice #": "Real-123",  # Simulate correct
        "Date": "2025-01-01",      # Simulate correct
        "Total Amount": "$150.00", # Simulate incorrect value
        "Vendor": None,           # Simulate not found / null
        "FieldToFail": "Some value" # Add a field designed to fail in mock
    }

    print("\n--- Testing review_fields (with mocked LLM) ---")
    from unittest.mock import patch
    with patch('llm_module.llm_call', mock_llm_review):
        review_results = review_fields(mock_text, mock_extracted)
        print("\n--- Final Review Results ---")
        print(json.dumps(review_results, indent=2))

    print("\n--- Testing parse_review directly ---")
    fields_for_parse_test = ["Invoice #", "Date", "Total Amount"]
    test_resp_review_ok = '{\n  "Invoice #": { "status": "PASS", "feedback": "-" },\n  "Date": { "status": "FAIL", "feedback": "Wrong format" },\n  "Total Amount": { "status": "PASS", "feedback": "" }\n}'
    print(f"Parsing good review JSON: {parse_review(test_resp_review_ok, fields_for_parse_test)}")
    test_resp_review_md = '```json\n{\n  "Invoice #": { "status": "PASS", "feedback": "Match" }\n}\n```'
    print(f"Parsing markdown review JSON: {parse_review(test_resp_review_md, ['Invoice #'])}")
    test_resp_review_bad_status = '{\n  "Invoice #": { "status": "PASSED", "feedback": "Typo" }\n}'
    print(f"Parsing review with bad status: {parse_review(test_resp_review_bad_status, ['Invoice #'])}")
    test_resp_review_no_json = 'This review failed.'
    print(f"Parsing review with no JSON: {parse_review(test_resp_review_no_json, ['Invoice #'])}") 