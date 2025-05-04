import os
from dotenv import load_dotenv
import openai
import anthropic

load_dotenv() # Load environment variables from .env file

def llm_call(prompt: str, *, model: str = None, use_anthropic: bool = None) -> str:
    """Makes a call to either OpenAI or Anthropic LLM based on environment variable or argument."""
    
    if use_anthropic is None:
        use_anthropic = os.getenv("USE_ANTHROPIC", "False").lower() in ('true', '1', 't')

    try:
        if use_anthropic:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-3-sonnet-20240229", # Example model, adjust as needed
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # Access the response content correctly based on Anthropic's structure
            if message.content and isinstance(message.content, list):
                return message.content[0].text
            else:
                 # Fallback or error handling if the structure is unexpected
                 print(f"Warning: Unexpected Anthropic response structure: {message.content}")
                 return ""
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini", # Example model, adjust as needed
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return resp.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Depending on the desired behavior, you might return an error message or re-raise
        return f"Error: Could not get response from LLM. Details: {e}"

# Simple test (optional)
if __name__ == "__main__":
    # Ensure you have a .env file with your keys in the same directory
    # or the keys are set as environment variables
    print("Testing OpenAI call:")
    print(llm_call("What is 2+2?", use_anthropic=False))

    # print("\nTesting Anthropic call:")
    # print(llm_call("What is the capital of France?", use_anthropic=True)) 