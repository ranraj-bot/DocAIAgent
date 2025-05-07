import json5
import base64
import os
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from llm_module import llm_call
from utils.extractor_utils import create_extraction_prompt_messages
from utils.reviewer_utils import create_review_prompt_messages
from utils.extractor_utils import parse_extraction
from utils.reviewer_utils import parse_review
from config import MODEL_CONFIG

load_dotenv()

# --- Custom Tool Definitions ---

@register_tool('extractor_ai')
class ExtractorAI(BaseTool):
    description = (
        "Extracts key-value pairs from a document given paths to its image data file (base64 encoded text) and OCR text file, "
        "a list of fields to extract, and the name of the LLM to use for extraction. "
        "The 'fields_to_extract' should be a JSON string array, e.g., '[\"field1\", \"field2\"]'."
    )
    parameters = [
        {
            'name': 'image_file_path',
            'type': 'string',
            'description': 'Path to the file containing the base64 encoded string of the image.',
            'required': True
        },
        {
            'name': 'ocr_file_path',
            'type': 'string',
            'description': 'Path to the file containing the OCR extracted text from the document.',
            'required': True
        },
        {
            'name': 'fields_to_extract',
            'type': 'string',
            'description': 'A JSON string array of fields to extract. Example: "[\"invoice_number\", \"total_amount\"]"',
            'required': True
        },
        {
            'name': 'llm_name',
            'type': 'string',
            'description': 'The conceptual name of the LLM to be used internally by this tool for extraction. Options are "qwen_25_vl" or "qwen_25". Prefer "qwen_25_vl" for its vision capabilities.`',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            tool_params = json5.loads(params)
            image_file_path = tool_params.get('image_file_path')
            ocr_file_path = tool_params.get('ocr_file_path')
            fields_to_extract_str = tool_params.get('fields_to_extract')
            llm_name = tool_params.get('llm_name')

            print(f"--- ExtractorAI called with LLM: {llm_name} ---")
            print(f"Reading image data from: {image_file_path}")
            print(f"Reading OCR text from: {ocr_file_path}")

            try:
                with open(image_file_path, 'r') as f:
                    image_bytes_b64 = f.read()
                img_bytes = base64.b64decode(image_bytes_b64) # Decode for use with create_extraction_prompt_messages
                with open(ocr_file_path, 'r') as f:
                    ocr_text = f.read()
            except Exception as e:
                print(f"Error reading image/OCR files in ExtractorAI: {e}")
                return json5.dumps({"error": f"Failed to read input files: {e}"}, ensure_ascii=False)

            print(f"ExtractorAI - OCR Text (first 100 chars): {ocr_text[:100]}")
            print(f"ExtractorAI - Fields to extract: {fields_to_extract_str}")

            # Actual LLM call for extraction
            try:
                extracted_fields = json5.loads(fields_to_extract_str)
                print(f"ExtractorAI - Parsed fields to extract: {extracted_fields}")

                messages = create_extraction_prompt_messages(extracted_fields, ocr_text, img_bytes)
                # The 'llm_name' parameter from the agent's system prompt is used as model_identifier
                response = llm_call(messages, model_identifier=llm_name) 
                extraction_result = parse_extraction(response, extracted_fields)
                
                print(f"ExtractorAI - LLM call successful, Raw LLM Response (first 100 chars): {str(response)[:100]}...")
                print(f"ExtractorAI - Parsed Extraction Result: {extraction_result}")
                return json5.dumps(extraction_result, ensure_ascii=False)

            except Exception as e:
                print(f"Error during ExtractorAI LLM call or parsing: {e}")
                return json5.dumps({"error": f"LLM call or parsing failed in ExtractorAI: {e}"}, ensure_ascii=False)

        except Exception as e:
            print(f"Error in ExtractorAI call: {e}")
            return json5.dumps({"error": str(e)}, ensure_ascii=False)

@register_tool('reviewer_ai')
class ReviewerAI(BaseTool):
    description = (
        "Reviews the extracted JSON data against the document by reading its image data file (base64 encoded text) and OCR text file. "
        "It uses a specified LLM for the review process and returns a review status for each field."
    )
    parameters = [
        {
            'name': 'image_file_path',
            'type': 'string',
            'description': 'Path to the file containing the base64 encoded string of the image.',
            'required': True
        },
        {
            'name': 'ocr_file_path',
            'type': 'string',
            'description': 'Path to the file containing the OCR extracted text from the document.',
            'required': True
        },
        {
            'name': 'extracted_json_str',
            'type': 'string',
            'description': 'The JSON string of data extracted by the ExtractorAI tool.',
            'required': True
        },
        {
            'name': 'llm_name',
            'type': 'string',
            'description': 'The conceptual name of the LLM to be used internally by this tool for review. Options are "qwen_25_vl" or "qwen_3".`',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            tool_params = json5.loads(params)
            image_file_path = tool_params.get('image_file_path')
            ocr_file_path = tool_params.get('ocr_file_path')
            extracted_json_str = tool_params.get('extracted_json_str')
            llm_name = tool_params.get('llm_name')

            print(f"--- ReviewerAI called with LLM: {llm_name} ---")
            print(f"Reading image data from: {image_file_path}")
            print(f"Reading OCR text from: {ocr_file_path}")
            print(f"Reading extracted JSON from: {extracted_json_str}")
            
            try:
                with open(image_file_path, 'r') as f:
                    image_bytes_b64 = f.read() 
                img_bytes = base64.b64decode(image_bytes_b64) # Decode for use with create_review_prompt_messages
                with open(ocr_file_path, 'r') as f:
                    ocr_text = f.read() 
            except Exception as e:
                print(f"Error reading image/OCR files in ReviewerAI: {e}")
                return json5.dumps({"error": f"Failed to read input files for review: {e}"}, ensure_ascii=False)

            # Actual LLM call for review
            try:
                extraction_result_dict = json5.loads(extracted_json_str)
                print(f"ReviewerAI - Parsed extracted JSON for review: {extraction_result_dict}")

                messages = create_review_prompt_messages(extraction_result_dict, ocr_text, img_bytes)
                # The 'llm_name' parameter from the agent's system prompt is used as model_identifier
                response = llm_call(messages, model_identifier=llm_name)
                # Assuming llm_call might return (response, reasoning) like in your notebook, 
                # but parse_review likely expects just the main response content.
                # Adjust if llm_call has a different signature in your actual module.
                if isinstance(response, tuple) and len(response) == 2:
                    actual_response_content = response[0] # Assuming first element is the main response
                else:
                    actual_response_content = response
                
                review_result = parse_review(actual_response_content, extraction_result_dict)
                
                print(f"ReviewerAI - LLM call successful, Raw LLM Response (first 100 chars): {str(actual_response_content)[:100]}...")
                print(f"ReviewerAI - Parsed Review Result: {review_result}")
                return json5.dumps(review_result, ensure_ascii=False)

            except Exception as e:
                print(f"Error during ReviewerAI LLM call or parsing: {e}")
                return json5.dumps({"error": f"LLM call or parsing failed in ReviewerAI: {e}"}, ensure_ascii=False)

        except Exception as e:
            print(f"Error in ReviewerAI call: {e}")
            return json5.dumps({"error": str(e)}, ensure_ascii=False)

# Determine the agent's LLM configuration dynamically
# Uses the 'qwen_3' configuration by default for the main agent LLM
AGENT_MODEL_KEY = "qwen_3" # Key to look up in MODEL_CONFIG and .env
agent_model_name = MODEL_CONFIG.get(AGENT_MODEL_KEY, {}).get("model_name")

env_key_prefix = AGENT_MODEL_KEY.upper().replace("/", "_")
agent_api_key_env = f"{env_key_prefix}_API_KEY"
agent_base_url_env = f"{env_key_prefix}_BASE_URL"
agent_api_key = os.getenv(agent_api_key_env)
agent_model_server = os.getenv(agent_base_url_env)

if not all([agent_model_name, agent_api_key, agent_model_server]):
    raise ValueError(
        f"Missing configuration for agent LLM ('{AGENT_MODEL_KEY}'). "
        f"Ensure '{AGENT_MODEL_KEY}' is in MODEL_CONFIG and corresponding "
        f"'{agent_api_key_env}' and '{agent_base_url_env}' are in .env file."
    )

LLM_CONFIG = {
    'model': agent_model_name,
    'model_server': agent_model_server,
    'api_key': agent_api_key,
}

# --- System Instruction for the Agent ---
SYSTEM_INSTRUCTION = (
    "You are a Document AI Agent. Your goal is to extract information from a document and have it reviewed. "
    "The user will provide you with the filenames for the document's image data (base64 encoded text) and its OCR text, "
    "and a list of fields they want to extract (as a JSON string array).\n"
    "The image data file and OCR text file are accessible by the tools using the file paths you provide them.\n"
    "Follow these steps:\n"
    "0. Decide which stage this turn is in. is it the extraction stage or the review stage?. If it is the first turn, it is the extraction stage. if there is extracted json output from the previous turn, it is the review stage. Only call one tool at a time.\n"
    "1. if it is the extraction stage:\n"
    "Call the `extractor_ai` tool. For its `llm_name` parameter, preffered llm_name is 'qwen_25_vl' (conceptually, for its vision capabilities). "
    "You MUST pass the `image_file_path` and `ocr_file_path` (as provided by the user or context) and `fields_to_extract` (as a JSON string e.g. '[\"field1\", \"field2\"]') to this tool.\n"
    "2. If it is the review stage, take the JSON output from `extractor_ai`.\n"
    "Call the `reviewer_ai` tool using the json output of `extractor_ai`."
    "You MUST pass the original `image_file_path`, `ocr_file_path`, and the `extracted_json_str` (output from step 1) to this tool.\n"
    "4. Finally, Combine the extracted data from `extractor_ai` and the review results from `reviewer_ai` to form the final output."
    "Output json should be of format:\n" 
    "FINAL_OUTPUT_JSON: { key: { value: <>, status: PASS/FAIL } }\n"
    "Present the final JSON to the user. DONOT print anything else after the FINAL_OUTPUT_JSON."
)