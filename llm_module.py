import os
from dotenv import load_dotenv
import openai
import traceback
from config import MODEL_CONFIG

load_dotenv()

def llm_call(messages: list[dict], model_identifier: str = "qwen_25_vl") -> str:
    """Makes a call to the configured OpenAI LLM based on a model identifier."""

    # --- Configuration Resolution ---
    if model_identifier not in MODEL_CONFIG:
        return f"Error: Invalid model_identifier '{model_identifier}'."

    config = MODEL_CONFIG[model_identifier]
    model_name = config.get("model_name")
    if not model_name:
        return f"Error: model_name not configured for '{model_identifier}'."

    print(f"Using model: {model_name}")

    # --- API Key & Endpoint Resolution ---
    api_key_var = f"{model_identifier.upper()}_API_KEY"
    api_key = os.getenv(api_key_var) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"Error: API Key missing ({api_key_var} or OPENAI_API_KEY)."

    base_url_var = f"{model_identifier.upper()}_BASE_URL"
    base_url = os.getenv(base_url_var) or None # Use None for default OpenAI URL
    # --- End Configuration Resolution ---

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        if "Qwen3" in model_name:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            return resp.choices[0].message.content, resp.choices[0].message.reasoning_content
        else:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1500,
                temperature=0
            )
            return resp.choices[0].message.content

    except Exception as e:
        print(f"--- ERROR calling OpenAI LLM (model: {model_name}) ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        # Return a concise error string for the UI/calling function
        return f"Error: LLM call failed ({type(e).__name__})."
