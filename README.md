# Document AI Agent
This Repo was for a workshop on How to build your own Document AI Agent using Open Source LLMs

## Overview

This project implements a Document AI Agent designed to process documents by performing (OCR), classifying document types, extracting key-value pairs using Large Language Models (LLMs), and reviewing the extracted information. It features a basic web interface built with Gradio and includes a Jupyter notebook for detailed component walkthroughs and model evaluations.

The agent orchestrates various LLMs, including vision-language models and reasoning models, to achieve accurate and reliable document processing.

## Features

*   **Document Upload:** Accepts image files (e.g., PNG, JPG, JPEG) for processing.
*   **OCR:** Extracts text from uploaded documents. Supports Google Vision OCR, Surya Ocr and Tessaract.
*   **Document Classification:** Identifies the type of document (e.g., invoice, receipt) and suggests relevant fields for extraction.
*   **Interactive Field Selection:** Allows users to confirm or modify the fields suggested by the classifier.
*   **LLM-Powered Extraction:** Utilizes LLMs (e.g., Qwen2.5-VL-7B-Instruct) to extract information for the selected fields.
*   **LLM-Powered Review:** Employs LLMs (text-only or multimodal, e.g., Qwen2.5-VL-7B-Instruct, Qwen3-8B) to review the accuracy of extracted data against the document.
*   **Gradio Web Interface:** Provides an intuitive UI for uploading documents, managing the processing workflow, and viewing results.
*   **Downloadable Results:** Allows users to download the final extracted and reviewed data in JSON format.
*   **Component-wise Walkthrough:** Includes a Jupyter Notebook (`Workshop_Walkthrough_Componenets (1).ipynb`) demonstrating the functionality of individual modules and comparing different LLM performances.

## Project Structure

The project is organized into several key modules:

*   `gradio_app.py`: The main Gradio application that provides the user interface and orchestrates the document processing workflow.
*   `document_ai_agent.py`: Defines the core agent logic using the `qwen-agent` framework, including custom tool definitions for extraction and reviewer.
*   `ocr_module.py`: Handles text extraction from documents.
*   `classifier.py`: Responsible for document type classification and suggesting fields.
*   `extractor.py`: Manages the extraction of key-value pairs using LLMs.
*   `reviewer.py`: Handles the review of extracted data using LLMs.
*   `llm_module.py`: Provides a centralized interface for making calls to various LLMs.
*   `utils/`: Contains utility functions for different modules (e.g., `extractor_utils.py`, `reviewer_utils.py`).
*   `config.py`: Stores configuration for different LLM models used in the project.
*   `.env`: Used for managing API keys and base URLs for LLM services (template provided in `.env.example` if available, otherwise structure can be inferred from code).
*   `requirements.txt`: Lists all Python dependencies.
*   `Workshop_Walkthrough_Componenets (1).ipynb`: A Jupyter notebook providing a detailed guide through the different components, showcasing their usage and model performance comparisons.

## Workflow

The typical workflow when using the Gradio application is as follows:

1.  **Upload Document:** The user uploads an image of the document.
2.  **OCR Processing:** The system performs OCR to extract raw text from the image.
3.  **Classification:** The extracted text (and potentially the image) is sent to a classifier LLM, which determines the document type and suggests a list of relevant fields.
4.  **Field Confirmation:** The user reviews the suggested fields and can modify the selection.
5.  **Extraction:** The selected fields, OCR text, and image are passed to an extractor LLM (often a multimodal model like Qwen2.5-VL) to extract the values.
6.  **Review (Optional but demonstrated):** The extracted data, OCR text, and image are passed to a reviewer LLM. This LLM verifies each extracted field against the document and provides a PASS/FAIL status and feedback. Different reviewer LLMs (text-only or multimodal) can be used.
7.  **Display Results:** The application displays the extracted values alongside their review status and feedback.
8.  **Download:** The user can download the final processed data as a JSON file.

The `document_ai_agent.py` implements a more autonomous agent flow where an LLM orchestrates calls to `extractor_ai` and `reviewer_ai` tools.

## Models Used

The project leverages several powerful Qwen models, configured in `config.py` and demonstrated in the notebook:

*   **Qwen2.5-7B-Instruct:** Used for tasks like classification (text-based).
*   **Qwen2.5-VL-7B-Instruct:** A vision-language model used for extraction and review, capable of processing both text and image information.
*   **Qwen/Qwen3-8B:** A more advanced text-based model, potentially used for reasoning-intensive review tasks.
*   Other models can be used by simply providing the api_base_url, api_key and model_name in .env and config.py

The specific model for each step (extraction, review, classification) is determined by the configuration and the logic within the respective modules or agent prompts.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    # git clone <repository_url>
    # cd <repository_name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure `pytesseract` is installed if you plan to use it as an OCR engine, which might require additional system-level Tesseract OCR dependencies. For Google Vision, ensure you have authenticated via `gcloud auth application-default login`.

4.  **Configure Environment Variables:**
    Create a `.gitignore` file in the root of your project (if one doesn't exist already) and add the following line to it, to ensure your actual secrets are not committed:
    ```gitignore
    .env
    ```
    Next, copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Then, open the newly created `.env` file and populate it with your actual API keys and base URLs for the LLM services you intend to use. The structure is based on `config.py` and `.env.example`:
    ```env
    # Example structure - fill with your actual values
    QWEN_25_API_KEY="YOUR_QWEN_25_API_KEY_HERE"
    QWEN_25_BASE_URL="your_llm_endpoint"
    QWEN_25_VL_API_KEY="your_api_key"
    QWEN_25_VL_BASE_URL="your_llm_endpoint"
    QWEN_3_API_KEY="your_api_key"
    QWEN_3_BASE_URL="your_llm_endpoint"
    QWEN_3_14_API_KEY="your_api_key"
    QWEN_3_14_BASE_URL="your_llm_endpoint"
    ```
    The application uses these variables to connect to different LLM backends. Ensure the LLM servers are running and accessible at the specified URLs.

## How to Run

1.  **Run the Gradio Web Application:**
    ```bash
    python gradio_app.py
    ```
    This will start the Gradio server, and you can access the UI in your web browser (typically at `http://127.0.0.1:7860` or a shared link if `share=True` is enabled).

2.  **Explore the Jupyter Notebook:**
    Launch Jupyter Lab or Jupyter Notebook and open `Workshop_Walkthrough_Componenets (1).ipynb` to understand the individual components, test different models, and see detailed examples of OCR, classification, extraction, review processes and Agentic Orchestration.
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```

## Hosting Local LLMs with vLLM (Optional)

If you prefer to host the language models locally, you can use [vLLM](https://github.com/vllm-project/vllm). vLLM is a fast and easy-to-use library for LLM inference and serving.

Here are example commands to serve some of the Qwen models used in this project:

*   **For Qwen3-8B (with reasoning capabilities):**
    ```bash
    vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1
    ```
    This command enables reasoning features which might be beneficial for the reviewer or agent components.

*   **For Qwen2.5-7B-Instruct:**
    ```bash
    vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --trust-remote-code
    ```
    Make sure to adjust model names, served-model-names, and ports as needed to match your `.env` configuration (e.g., `QWEN_3_BASE_URL="http://localhost:8000/v1"` if vLLM serves on port 8000 by default).

Refer to the [vLLM documentation](https://vllm.readthedocs.io/en/latest/index.html) for more details on installation, supported models, and serving options.

## Future Enhancements (Inferred from Notebook)

*   **Visual Reasoning:** The notebook mentions "Future: Visual Reasoning," indicating plans or potential for incorporating more advanced visual understanding capabilities into the review or extraction processes.
*   **Agent-based Processing:** The `document_ai_agent.py` demonstrates an agentic approach which could be further developed for more complex, multi-step document processing tasks and added to the UI. Demo is availible in the Jupyter notebook

---