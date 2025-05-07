import gradio as gr
import pandas as pd
import json
import time
from io import BytesIO
from PIL import Image
import os # For potential temporary file handling if needed
import traceback # Import the traceback module

# --- Import Core Logic Modules ---
# Ensure these modules and the .env file are in the same directory or accessible
try:
    from ocr_module import extract_text
    from classifier import classify_and_suggest_fields
    from extractor import extract_key_value_pairs # Ensure this matches the refactored name
    from reviewer import review_fields # Ensure this matches the refactored name
    # Load API keys if llm_module uses dotenv
    from dotenv import load_dotenv
    load_dotenv()
    print("Core logic modules loaded successfully.")
except ImportError as e:
    print("--- ERROR IMPORTING CORE MODULES --- ") # Add header for clarity
    traceback.print_exc() # *** THIS PRINTS THE FULL TRACEBACK ***
    print(f"Error details: {e}") # Keep original error message
    print("--- DEFINING DUMMY FUNCTIONS AS FALLBACK --- ") # Add footer
    # Define dummy functions if modules fail to import, allowing UI to load
    def extract_text(img_bytes, ocr_engine='google_vision'): return f"DUMMY OCR for {len(img_bytes)} bytes. Import failed."
    def classify_and_suggest_fields(text, user_fields=None): return {"doc_type": "dummy (import failed)", "fields": ["Field A", "Field B"]}
    def extract_key_value_pairs(fields, ocr_text="", image_bytes=None): return {f: f"dummy value {i+1}" for i, f in enumerate(fields)}
    def review_fields(fields_to_review, ocr_text="", image_bytes=None): return {f: {"status": "DUMMY", "feedback": "import failed"} for f in fields_to_review}


# --- Constants ---
INITIAL_STATUS = "**Status:** Idle - Waiting for document upload."
DEFAULT_FIELDS = ["Field A", "Field B", "Field C"] # Fallback if classification fails badly


# --- Gradio Event Handlers ---

def handle_upload(file_obj):
    """Handles file upload, triggers OCR and Classification sequentially."""
    if file_obj is None:
        yield {
            status_text: gr.update(value=INITIAL_STATUS),
            image_display: gr.update(value=None, visible=False),
            field_selection_group: gr.update(visible=False),
            extraction_display: gr.update(value=None, visible=False),
            review_display: gr.update(value=None, visible=False),
            download_button: gr.update(visible=False),
            # State updates (Corrected: Return value directly)
            state_step: 0,
            state_image_bytes: None,
            state_ocr_text: None,
            state_classification_result: None,
            state_selected_fields: [],
            state_extraction_result: None,
            state_review_result: None,
            state_filename: None,
        }
        return # Exit if no file object

    # --- MODIFIED FILE READING ---
    file_path = file_obj
    if not isinstance(file_path, str) or not os.path.exists(file_path):
         print(f"Error: handle_upload received unexpected object type or invalid path: {type(file_obj)}, value: {file_obj}")
         yield { status_text: gr.update(value="**Status:** Error - Invalid file object received.") }
         return

    filename = os.path.basename(file_path)
    print(f"File uploaded: {filename}, Path: {file_path}")
    try:
        with open(file_path, 'rb') as f: # Open the file at the path
            img_bytes = f.read() # Read bytes from the opened file
    except Exception as e:
        print(f"Error reading file from path {file_path}: {e}")
        yield { status_text: gr.update(value=f"**Status:** Error reading uploaded file: {e}") }
        return
    # --- END MODIFIED FILE READING ---
    
    # Update UI immediately: Show image, update status
    yield {
        status_text: gr.update(value="**Status:** Processing - Performing OCR... 👁️"),
        image_display: gr.update(value=Image.open(BytesIO(img_bytes)), visible=True), # Display image
        state_image_bytes: img_bytes, # Corrected
        state_filename: filename,     # Corrected
        state_step: 1,                # Corrected
        # Hide previous results if any
        field_selection_group: gr.update(visible=False),
        extraction_display: gr.update(visible=False),
        review_display: gr.update(visible=False),
        download_button: gr.update(visible=False),
    }

    # --- OCR Step ---
    try:
        print("Performing OCR...")
        ocr_text_result = extract_text(img_bytes, ocr_engine='google_vision')
        #print(f"OCR Text (first 100 chars): {ocr_text_result[:100]}")
        if "ERROR" in ocr_text_result:
            raise ValueError(f"OCR Failed: {ocr_text_result}")
        yield { state_ocr_text: ocr_text_result }
    except Exception as e:
        print(f"Error during OCR: {e}")
        yield { status_text: gr.update(value=f"**Status:** Error during OCR: {e}") }
        return # Stop processing

    # --- Classification Step ---
    yield { status_text: gr.update(value="**Status:** Processing - Classifying document... 🧠") }
    try:
        print("Classifying document...")
        classification_res = classify_and_suggest_fields(
            text=ocr_text_result, 
            image_bytes=img_bytes
        )
        print(f"Classification Result: {classification_res}")
        doc_type = classification_res.get("doc_type", "error")
        suggested_fields = classification_res.get("fields", [])

        if doc_type == "error" or not suggested_fields:
             print("Classification failed or returned no fields, using defaults.")
             doc_type = "unknown (using defaults)"
             suggested_fields = DEFAULT_FIELDS # Use defaults
             classification_res = {"doc_type": doc_type, "fields": suggested_fields} # Update result state

        # Update UI for field selection
        yield {
            status_text: gr.update(value="**Status:** Action Required - Select fields below and click confirm."),
            field_doc_type: gr.update(value=f"**Detected Document Type:** {doc_type.capitalize()}"),
            field_checkboxes: gr.update(choices=suggested_fields, value=suggested_fields, interactive=True),
            confirm_button: gr.update(interactive=True),
            field_selection_group: gr.update(visible=True),
            # State updates
            state_classification_result: classification_res, # Corrected
            state_selected_fields: suggested_fields,       # Corrected
            state_step: 3,                                 # Corrected
        }

    except Exception as e:
        print(f"Error during Classification: {e}")
        yield { status_text: gr.update(value=f"**Status:** Error during Classification: {e}") }
        # Decide how to handle - maybe show default fields?
        yield {
             status_text: gr.update(value="**Status:** Classification Failed - Using default fields. Select and confirm."),
             field_doc_type: gr.update(value=f"**Detected Document Type:** Error"),
             field_checkboxes: gr.update(choices=DEFAULT_FIELDS, value=DEFAULT_FIELDS, interactive=True),
             confirm_button: gr.update(interactive=True),
             field_selection_group: gr.update(visible=True),
             state_classification_result: {"doc_type": "error", "fields": DEFAULT_FIELDS}, # Corrected
             state_selected_fields: DEFAULT_FIELDS, # Corrected
             state_step: 3, # Corrected
         }


def handle_confirm_and_extract(selected_fields_list, ocr_text, image_bytes):
    """Handles field confirmation, triggers extraction, pause, and review."""
    if not selected_fields_list:
        # This shouldn't happen if button is disabled, but as a fallback
        yield { status_text: gr.update(value="**Status:** Error - No fields selected.") }
        return

    print(f"Fields confirmed: {selected_fields_list}. Starting extraction.")

    # --- Extraction Step ---
    yield {
        status_text: gr.update(value="**Status:** Processing - Extracting fields... ✍️"),
        field_selection_group: gr.update(visible=False), # Hide selection UI
        confirm_button: gr.update(interactive=False), # Disable button
        field_checkboxes: gr.update(interactive=False), # Disable checkboxes
        extraction_display: gr.update(visible=False), # Hide old results
        review_display: gr.update(visible=False),
        download_button: gr.update(visible=False),
        state_selected_fields: selected_fields_list, # Corrected
        state_step: 4,                             # Corrected
    }

    try:
        extraction_res = extract_key_value_pairs(selected_fields_list, ocr_text, image_bytes=image_bytes)
        print(f"Extraction result: {extraction_res}")

        # Prepare DataFrame for display
        if extraction_res:
            df_extract = pd.DataFrame(list(extraction_res.items()), columns=['Field', 'Extracted Value'])
        else:
            df_extract = pd.DataFrame(columns=['Field', 'Extracted Value']) # Empty DF if needed
            print("Warning: Extraction result was empty.")

        yield {
            status_text: gr.update(value="**Status:** Extraction Complete. Pausing before review..."),
            extraction_display: gr.update(value=df_extract, visible=True),
            state_extraction_result: extraction_res, # Corrected
            state_step: 5,                         # Corrected
        }
    except Exception as e:
        print(f"Error during Extraction: {e}")
        yield { status_text: gr.update(value=f"**Status:** Error during Extraction: {e}") }
        return # Stop processing

    # --- Pause Step ---
    print("Pausing for 2 seconds...")
    time.sleep(2)
    yield { state_step: 6 } # Corrected: Update step before review starts

    # --- Review Step ---
    yield { status_text: gr.update(value="**Status:** Processing - Reviewing extraction... 🧐") }
    try:
        print("Reviewing extracted fields...")
        review_res = review_fields(
            fields_to_review=extraction_res, 
            ocr_text=ocr_text, 
            image_bytes=image_bytes
        )
        print(f"Review result: {review_res}")

        # Prepare final DataFrame with review results
        display_data = []
        # Ensure extraction_res is a dict before iterating its keys for review display
        effective_fields_to_review = list(extraction_res.keys()) if isinstance(extraction_res, dict) else selected_fields_list
        
        for field in effective_fields_to_review: 
            extracted_value = extraction_res.get(field, "Not Extracted") if isinstance(extraction_res, dict) else "Extraction Error"
            review = review_res.get(field, {"status": "ERROR", "feedback": "Not reviewed"}) if isinstance(review_res, dict) else {"status": "ERROR", "feedback": "Review Error"}
            status_icon = "✅" if review.get("status") == "PASS" else ("❌" if review.get("status") == "FAIL" else "❓")
            display_data.append({
                'Field': field,
                'Extracted Value': extracted_value,
                'Status': f'{status_icon} {review.get("status")}',
                'Feedback': review.get("feedback", "")
            })
        df_review = pd.DataFrame(display_data)

        yield {
            status_text: gr.update(value="**Status:** Complete - Review finished. ✅"),
            extraction_display: gr.update(visible=False), 
            review_display: gr.update(value=df_review, visible=True), 
            download_button: gr.update(visible=True, interactive=True), 
            state_review_result: review_res, 
            state_step: 7, 
        }
    except Exception as e:
        print(f"Error during Review: {e}")
        yield { status_text: gr.update(value=f"**Status:** Error during Review: {e}") }
        # Show extraction result again maybe?
        yield { extraction_display: gr.update(visible=True) }


def prepare_download_json(extraction_result):
    """Creates a JSON file from the extraction results for download."""
    if not extraction_result:
        print("No extraction data to download.")
        # Gradio DownloadButton expects a file path or None
        return None
    
    try:
        json_string = json.dumps(extraction_result, indent=2)
        # Use BytesIO to avoid saving a physical file if possible with DownloadButton
        json_bytes = BytesIO(json_string.encode('utf-8'))
        # It seems DownloadButton needs a file path, let's create a temp one
        # Ideally use tempfile module
        temp_file_path = "temp_extracted_data.json"
        with open(temp_file_path, "w", encoding="utf-8") as f:
             f.write(json_string)
        print(f"Prepared download file: {temp_file_path}")
        # The DownloadButton component itself will handle serving this path
        return temp_file_path
    except Exception as e:
        print(f"Error preparing download data: {e}")
        return None


def reset_all():
    """Returns updates to reset all components and state."""
    print("Resetting Gradio UI and State.")
    # Use gr.update for UI components, direct value for state
    return {
        status_text: gr.update(value=INITIAL_STATUS),
        image_display: gr.update(value=None, visible=False),
        field_doc_type: gr.update(value=""),
        field_checkboxes: gr.update(choices=[], value=[], interactive=False),
        confirm_button: gr.update(interactive=False),
        field_selection_group: gr.update(visible=False),
        extraction_display: gr.update(value=None, visible=False),
        review_display: gr.update(value=None, visible=False),
        download_button: gr.update(visible=False, value=None), # Reset file path too
        upload_button: gr.update(value=None), # Clear the upload button
        # State updates (Corrected)
        state_step: 0,
        state_image_bytes: None,
        state_ocr_text: None,
        state_classification_result: None,
        state_selected_fields: [],
        state_extraction_result: None,
        state_review_result: None,
        state_filename: None,
    }


# --- Gradio Interface Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="🕵️ Document AI Agent Demo") as app:
    gr.Markdown("# 🕵️ Document AI Agent Demo")

    # --- State Variables ---
    # Store internal state without displaying it directly
    state_step = gr.State(value=0)
    state_image_bytes = gr.State(value=None)
    state_ocr_text = gr.State(value=None)
    state_classification_result = gr.State(value=None) # Stores {doc_type: ..., fields: [...]}
    state_selected_fields = gr.State(value=[])
    state_extraction_result = gr.State(value=None) # Stores {field: value, ...}
    state_review_result = gr.State(value=None) # Stores {field: {status: ..., feedback: ...}, ...}
    state_filename = gr.State(value=None)

    # --- UI Layout ---
    with gr.Row():
        # --- Left Column ---
        with gr.Column(scale=6):
            status_text = gr.Markdown(value=INITIAL_STATUS)

            with gr.Row():
                upload_button = gr.UploadButton(
                    "Click to Upload Document",
                    file_types=["image"], # ["png", "jpg", "jpeg"]
                    # label="Upload Document Image",
                    scale=3 # Give more space
                )
                reset_button = gr.Button("Reset Agent", scale=1)

            gr.Markdown("---") # Divider

            # Main area where dynamic content appears
            with gr.Column(visible=True) as main_area: # Always visible column
                # Group for classification/selection UI elements
                with gr.Group(visible=False) as field_selection_group:
                    field_doc_type = gr.Markdown("**Detected Document Type:**")
                    field_checkboxes = gr.CheckboxGroup(
                        label="Select/Confirm Fields for Extraction",
                        interactive=False # Start disabled
                    )
                    confirm_button = gr.Button(
                        "Confirm Fields & Start Extraction",
                        variant="primary",
                        interactive=False # Start disabled
                    )

                # Displays for results
                extraction_display = gr.DataFrame(
                    label="Extracted Values",
                    headers=["Field", "Extracted Value"],
                    visible=False,
                    interactive=False
                )
                review_display = gr.DataFrame(
                    label="Review & Feedback Results",
                    headers=["Field", "Extracted Value", "Status", "Feedback"],
                    visible=False,
                    interactive=False
                )
                # Use gr.File for download flexibility? Or DownloadButton + helper func.
                download_button = gr.DownloadButton(
                    label="⬇️ Download Extracted JSON",
                    visible=False,
                    interactive=False
                )


        # --- Right Column ---
        with gr.Column(scale=4):
            image_display = gr.Image(
                label="Uploaded Document Preview",
                type="pil", # Read as PIL image for display
                visible=False, # Start hidden
                interactive=False,
                height=600 # Adjust height as needed
            )

    # --- Event Wiring ---
    
    # Define outputs for each handler
    upload_outputs = [
        status_text, image_display, field_selection_group, extraction_display, review_display,
        download_button, field_doc_type, field_checkboxes, confirm_button,
        state_step, state_image_bytes, state_ocr_text, state_classification_result,
        state_selected_fields, state_extraction_result, state_review_result, state_filename
    ]
    extract_outputs = [
        status_text, field_selection_group, confirm_button, field_checkboxes,
        extraction_display, review_display, download_button,
        state_step, state_selected_fields, state_extraction_result, state_review_result
    ]
    reset_outputs = [
        status_text, image_display, field_doc_type, field_checkboxes, confirm_button,
        field_selection_group, extraction_display, review_display, download_button, upload_button,
        state_step, state_image_bytes, state_ocr_text, state_classification_result,
        state_selected_fields, state_extraction_result, state_review_result, state_filename
    ]

    # --- Triggering the Flow ---
    
    # 1. Upload triggers the main processing function
    upload_button.upload(
        fn=handle_upload,
        inputs=[upload_button],
        outputs=upload_outputs,
        show_progress="hidden" # Hide default Gradio progress bar, we use status text
    )

    # 2. Confirm button triggers extraction and subsequent review
    confirm_button.click(
        fn=handle_confirm_and_extract,
        inputs=[field_checkboxes, state_ocr_text, state_image_bytes],
        outputs=extract_outputs,
         show_progress="hidden"
    )
    
    # 3. Download button needs the extraction result to prepare the file
    # Note: DownloadButton triggers the function *when clicked*, func must return a file path
    download_button.click(
        fn=prepare_download_json,
        inputs=[state_extraction_result],
        outputs=download_button # The button component itself receives the file path
    )

    # 4. Reset button clears everything
    reset_button.click(
        fn=reset_all,
        inputs=None,
        outputs=reset_outputs,
         show_progress="hidden"
    )


# --- Launch the App ---
if __name__ == "__main__":
    app.launch(debug=True, share=True) # Enable debug for easier troubleshooting
    # Remember to remove debug=True for production