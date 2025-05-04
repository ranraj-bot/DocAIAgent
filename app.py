import streamlit as st
import pandas as pd
import json
from io import BytesIO

# Import our modules
from ocr_module import extract_text
from classifier import classify_and_suggest_fields
from extractor import extract_fields
from reviewer import review_fields

# --- Page Configuration ---
st.set_page_config(
    page_title="🕵️ Document AI Agent Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🕵️ Document AI Agent Demo")
st.caption("Upload an image (PNG, JPG) containing text like an invoice or receipt.")

# --- Session State Initialization ---
# Initialize state variables to track the process flow and store data
# Step definitions:
# 0: Initial state, waiting for upload
# 1: File uploaded, ready for OCR
# 2: OCR done, ready for Classification
# 3: Classification done, show field selection UI
# 4: Fields selected/confirmed, ready for Extraction
# 5: Extraction done, show extraction results, ready for Review
# 6: Review processing started/in progress
# 7: Review done, show review results

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'selected_fields' not in st.session_state:
    st.session_state.selected_fields = []
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'review_result' not in st.session_state:
    st.session_state.review_result = None

# --- Helper Functions ---
def reset_app_state():
    """Resets the session state to the initial values."""
    keys_to_reset = [
        'step', 'uploaded_file_name', 'image_bytes', 'ocr_text',
        'classification_result', 'selected_fields', 'extraction_result',
        'review_result'
    ]
    # Set default values
    st.session_state.step = 0
    st.session_state.uploaded_file_name = None
    st.session_state.image_bytes = None
    st.session_state.ocr_text = None
    st.session_state.classification_result = None
    st.session_state.selected_fields = []
    st.session_state.extraction_result = None
    st.session_state.review_result = None

    print("Resetting app state.")
    # Clear the file uploader explicitly if possible (using key)
    # st.session_state.file_uploader = None # This might not work reliably depending on Streamlit version
    st.rerun()

# --- UI Layout ---

# Column for Controls (Upload, Reset)
col1, col2 = st.columns([0.7, 0.3])

with col1:
    # --- Step 1: File Upload ---
    with st.container(border=True):
        st.subheader("[1] Upload Document")
        uploaded_file_obj = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="file_uploader" # Key helps maintain widget state somewhat ff ff
        )

        # --- File Handling Logic ---
        if uploaded_file_obj is not None:
            # Check if this filename is NEW compared to what's in state, OR if we are in initial state 0
            is_new_file = (uploaded_file_obj.name != st.session_state.get('uploaded_file_name', None))
            
            if is_new_file:
                print(f"New file detected: {uploaded_file_obj.name}. Processing.")
                # Manually reset downstream state for the new file, keep step 0 for now
                st.session_state.image_bytes = None
                st.session_state.ocr_text = None
                st.session_state.classification_result = None
                st.session_state.selected_fields = []
                st.session_state.extraction_result = None
                st.session_state.review_result = None
                
                # Now store the new file info and set step to start processing
                st.session_state.uploaded_file_name = uploaded_file_obj.name
                st.session_state.image_bytes = uploaded_file_obj.getvalue()
                st.session_state.step = 1 # Set step to 1 AFTER storing file info
                print(f"State updated for {uploaded_file_obj.name}. Rerunning to start OCR.")
                st.rerun() # Rerun to start the processing pipeline at Step 1 (OCR)

            # If it's not a new file name, and we have image bytes, display the image.
            # This covers subsequent reruns where the same file is still in the uploader.
            elif st.session_state.image_bytes:
                 # Display image, but don't change state or rerun
                 st.image(st.session_state.image_bytes, caption=f"Processing: {st.session_state.uploaded_file_name}", width=300)

        elif st.session_state.get('uploaded_file_name') is not None:
            # Uploader is empty, but we had a filename -> User cleared the file. Reset everything.
            print("File removed by user.")
            # Call the full reset function only when the file is explicitly cleared
            reset_app_state() # This function includes its own rerun

with col2:
    st.button("Reset Agent", on_click=reset_app_state, use_container_width=True, disabled=(st.session_state.step == 0))

# --- Main Processing Area (Sequential Steps) ---

# Step 1 -> 2: Perform OCR
if st.session_state.step == 1 and st.session_state.image_bytes:
    print("Step 1 -> 2: Performing OCR")
    with st.spinner("👁️ Performing OCR..."):
        st.session_state.ocr_text = extract_text(st.session_state.image_bytes)
        print(f"OCR Complete. Text length: {len(st.session_state.ocr_text)}")
        if "ERROR" in st.session_state.ocr_text:
            st.error(f"OCR Failed: {st.session_state.ocr_text}")
            st.stop()
        st.session_state.step = 2
        st.rerun()

# Step 2 -> 3: Perform Classification
if st.session_state.step == 2 and st.session_state.ocr_text:
    print("Step 2 -> 3: Classifying document")
    # Display container for classification (will show spinner)
    with st.container(border=True):
        st.subheader("[2] Document Classification & Field Selection")
        with st.spinner("🧠 Classifying document and suggesting fields..."):
            st.session_state.classification_result = classify_and_suggest_fields(st.session_state.ocr_text)
            print(f"Classification Result: {st.session_state.classification_result}")
            if st.session_state.classification_result.get("doc_type") == "error":
                st.error("Failed to classify the document.")
                st.stop()
            # Default selected fields to suggested fields
            st.session_state.selected_fields = st.session_state.classification_result.get("fields", [])
            st.session_state.step = 3
            st.rerun()

# Step 3: Display Classification & Allow Field Selection
if st.session_state.step == 3:
    print("Step 3: Displaying Classification & Field Selection UI")
    with st.container(border=True):
        st.subheader("[2] Document Classification & Field Selection")
        if st.session_state.classification_result:
            doc_type = st.session_state.classification_result.get("doc_type", "Unknown")
            suggested_fields = st.session_state.classification_result.get("fields", [])
            st.write(f"**Detected Document Type:** {doc_type.capitalize()}")

            st.write("**Select Fields for Extraction:**")
            cols = st.columns(3)
            current_selection = []
            # Use stored selected_fields for default checkbox state
            selected_in_state = st.session_state.get('selected_fields', [])
            for i, field in enumerate(suggested_fields):
                with cols[i % 3]:
                    # Key ensures widget identity across runs
                    is_selected = st.checkbox(field, value=(field in selected_in_state), key=f"cb_{field}")
                    if is_selected:
                        current_selection.append(field)

            # Button to confirm selection and proceed
            proceed_to_extract = st.button("Confirm Fields & Extract", key="confirm_extract")

            if proceed_to_extract:
                if not current_selection:
                    st.warning("Please select at least one field.")
                else:
                    print(f"Fields confirmed: {current_selection}. Proceeding to extraction.")
                    st.session_state.selected_fields = current_selection
                    # Clear downstream results if selection is re-confirmed
                    st.session_state.extraction_result = None
                    st.session_state.review_result = None
                    st.session_state.step = 4 # Set step to trigger extraction on next run
                    st.rerun()
        else:
            st.warning("Classification result not available.")

# Step 4 -> 5: Perform Extraction
if st.session_state.step == 4 and st.session_state.ocr_text and st.session_state.selected_fields:
    print("Step 4 -> 5: Extracting fields")
    # Display container for extraction (will show spinner)
    with st.container(border=True):
        st.subheader("[3] Field Extraction")
        with st.spinner("✍️ Extracting selected fields..."):
            st.session_state.extraction_result = extract_fields(
                st.session_state.ocr_text,
                st.session_state.selected_fields
            )
            print(f"Extraction Result: {st.session_state.extraction_result}")
            if not st.session_state.extraction_result or not any(st.session_state.extraction_result.values()):
                 st.warning("Extraction might have failed or found no values.")
            st.session_state.step = 5 # Extraction done, ready to display results
            st.rerun()

# Step 5: Display Extraction Results
if st.session_state.step == 5:
    print("Step 5: Displaying Extraction Results")
    with st.container(border=True):
        st.subheader("[3] Field Extraction")
        if st.session_state.extraction_result is not None:
            st.write("**Extracted Values:**")
            df_extract = pd.DataFrame(
                list(st.session_state.extraction_result.items()),
                columns=['Field', 'Extracted Value']
            )
            st.dataframe(df_extract, use_container_width=True, hide_index=True)

            # Automatically proceed to review after displaying extraction
            print("Triggering review step.")
            st.session_state.step = 6 # Ready for review processing
            st.rerun()
        else:
             st.write("No extraction data available.")


# Step 6 -> 7: Perform Review
if st.session_state.step == 6 and st.session_state.ocr_text and st.session_state.extraction_result is not None:
    print("Step 6 -> 7: Reviewing extracted fields")
    # Display container for review (will show spinner)
    with st.container(border=True):
        st.subheader("[4] Review & Feedback")
        with st.spinner("🧐 Reviewing extracted fields..."):
            st.session_state.review_result = review_fields(
                st.session_state.ocr_text,
                st.session_state.extraction_result
            )
            print(f"Review Result: {st.session_state.review_result}")
            st.session_state.step = 7 # Review done, ready for final display
            st.rerun()

# Step 7: Display Review Results
if st.session_state.step == 7:
    print("Step 7: Displaying Review Results")
    with st.container(border=True):
        st.subheader("[4] Review & Feedback")
        if st.session_state.review_result is not None and st.session_state.extraction_result is not None:
            st.write("**Review Status:**")
            display_data = []
            for field, extracted_value in st.session_state.extraction_result.items():
                review = st.session_state.review_result.get(field, {"status": "ERROR", "feedback": "Not reviewed"})
                status_icon = "✅" if review.get("status") == "PASS" else ("❌" if review.get("status") == "FAIL" else "❓")
                display_data.append({
                    'Field': field,
                    'Extracted Value': extracted_value,
                    'Status': f'{status_icon} {review.get("status")}',
                    'Feedback': review.get("feedback", "")
                })
            df_review = pd.DataFrame(display_data)
            st.dataframe(df_review, use_container_width=True, hide_index=True)

            # Download button
            try:
                final_json = json.dumps(st.session_state.extraction_result, indent=2)
                file_name = f"{st.session_state.uploaded_file_name}_extracted.json" if st.session_state.uploaded_file_name else "extracted.json"
                st.download_button(
                    label="⬇️ Download Extracted JSON",
                    data=final_json,
                    file_name=file_name,
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error preparing download button: {e}")

        else:
             st.warning("Review or extraction results not available for display.")

# --- Footer/Debug Info (Optional) ---
# st.divider()
# with st.expander("Debug: Show Session State"):
#    st.write(st.session_state)