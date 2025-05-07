# Central configuration settings for the Document AI Agent.
MODEL_CONFIG = {
    "extractor": {
        "model_name": "Qwen2.5-VL-7B-Instruct" # Vision-Language Model
    },
    "reviewer": {
        "model_name": "Qwen2.5-VL-7B-Instruct" # Model used for review
    },
    "classifier": {
       "model_name": "Qwen2.5-7B-Instruct" # Text LLM
    },
    "qwen_25": {
        "model_name": "Qwen2.5-7B-Instruct"
    },
    "qwen_25_vl": {
        "model_name": "Qwen2.5-VL-7B-Instruct"
    },
    "qwen_3": {
        "model_name": "Qwen/Qwen3-8B"
    },
    "qwen_3_14": {
        "model_name": "Qwen/Qwen3-14B"
    }
}
