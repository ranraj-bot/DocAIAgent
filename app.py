import streamlit as st
import pandas as pd
import json
import time # Import time for the pause
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
# Removed caption, status bar will provide context

# --- Session State Initialization ---
# Step definitions refined for the new flow:
# 0: Initial state, waiting for upload
# 1: File uploaded, ready for OCR
# 2: OCR done, ready for Classification
# 3: Classification done, showing field selection UI (waiting for user confirmation)
# 4: Fields confirmed by user, ready for Extraction
# 5: Extraction done, showing extraction results (briefly before review)
# 6: Pausing before Review / Triggering Review
# 7: Review done, showing review results

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
    st.session_state.step = 0
    st.session_state.uploaded_file_name = None
    st.session_state.image_bytes = None
    st.session_state.ocr_text = None
    st.session_state.classification_result = None
    st.session_state.selected_fields = []
    st.session_state.extraction_result = None
    st.session_state.review_result = None
    print("Resetting app state.")
    st.rerun()

# --- UI Layout: Two Columns ---
left_column, right_column = st.columns([0.6, 0.4]) # Adjust ratio as needed

# --- Right Column (Context: Image Display) ---
with right_column:
    st.subheader("Uploaded Document")
    if st.session_state.image_bytes:
        try:
            st.image(st.session_state.image_bytes, caption=f"Context: {st.session_state.uploaded_file_name}", use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Image will appear here once uploaded.")

# --- Left Column (Agent Workspace: Status & Interaction) ---
with left_column:
    # --- Persistent Status Bar ---
    status_placeholder = st.empty()

    # --- Compact Upload & Reset Area ---
    upload_col, reset_col = st.columns([0.7, 0.3])
    with upload_col:
        # Only show uploader if no file is loaded yet (step 0)
        if st.session_state.step == 0:
            uploaded_file_obj = st.file_uploader(
                "Upload Document Image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed", # Use button-like appearance
                key="file_uploader"
            )
            if uploaded_file_obj is not None:
                print(f"File uploaded: {uploaded_file_obj.name}. Storing bytes.")
                st.session_state.uploaded_file_name = uploaded_file_obj.name
                st.session_state.image_bytes = uploaded_file_obj.getvalue()
                st.session_state.step = 1 # Move to OCR step
                st.rerun()
        else:
            # Show filename once loaded
             if st.session_state.uploaded_file_name:
                 st.write(f"**File:** {st.session_state.uploaded_file_name}")


    with reset_col:
        st.button("Reset Agent", on_click=reset_app_state, use_container_width=True, disabled=(st.session_state.step == 0))

    st.divider() # Separator

    # --- Dynamic Main Interaction Area ---
    main_area_placeholder = st.container()

    # --- Processing Logic and Status Updates ---

    current_step = st.session_state.step

    # Update Status Bar based on current step BEFORE processing for that step
    if current_step == 0:
        status_placeholder.info("**Status:** Idle - Waiting for document upload.")
    elif current_step == 1:
        status_placeholder.info("**Status:** Processing - Performing OCR... 👁️")
    elif current_step == 2:
        status_placeholder.info("**Status:** Processing - Classifying document... 🧠")
    elif current_step == 3:
        status_placeholder.warning("**Status:** Action Required - Select fields below and click confirm.")
    elif current_step == 4:
        status_placeholder.info("**Status:** Processing - Extracting fields... ✍️")
    elif current_step == 5:
        status_placeholder.success("**Status:** Extraction Complete. Preparing for review...")
    elif current_step == 6:
         status_placeholder.info("**Status:** Processing - Reviewing extraction... 🧐")
    elif current_step == 7:
        status_placeholder.success("**Status:** Complete - Review finished. ✅")


    # --- Step Execution and UI Display within Main Area ---

    # Step 1 -> 2: Perform OCR
    if current_step == 1 and st.session_state.image_bytes:
        print("Step 1 -> 2: Performing OCR")
        try:
            # Display spinner within the status placeholder temporarily
            with status_placeholder:
                 with st.spinner("👁️ Performing OCR..."):
                     ocr_text_result = extract_text(st.session_state.image_bytes)
            
            print(f"OCR Complete. Text length: {len(ocr_text_result)}")
            if "ERROR" in ocr_text_result:
                 st.error(f"OCR Failed: {ocr_text_result}")
                 reset_app_state() # Reset on critical error
            else:
                st.session_state.ocr_text = ocr_text_result
                st.session_state.step = 2
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred during OCR: {e}")
            reset_app_state()

    # Step 2 -> 3: Perform Classification
    elif current_step == 2 and st.session_state.ocr_text:
        print("Step 2 -> 3: Classifying document")
        try:
            with status_placeholder:
                 with st.spinner("🧠 Classifying document..."):
                    classification_res = classify_and_suggest_fields(st.session_state.ocr_text)

            print(f"Classification Result: {classification_res}")
            if classification_res.get("doc_type") == "error":
                st.error("Failed to classify the document.")
                reset_app_state()
            else:
                st.session_state.classification_result = classification_res
                st.session_state.selected_fields = st.session_state.classification_result.get("fields", [])
                st.session_state.step = 3
                st.rerun()
        except Exception as e:
             st.error(f"An error occurred during classification: {e}")
             reset_app_state()

    # Step 3: Display Classification & Allow Field Selection
    elif current_step == 3:
        print("Step 3: Displaying Classification & Field Selection UI")
        with main_area_placeholder:
            st.subheader("Document Classification & Field Selection")
            if st.session_state.classification_result:
                doc_type = st.session_state.classification_result.get("doc_type", "Unknown")
                suggested_fields = st.session_state.classification_result.get("fields", [])
                st.write(f"**Detected Document Type:** {doc_type.capitalize()}")

                st.write("**Select/Confirm Fields for Extraction:**")
                # Use a form to group checkboxes and the button
                with st.form(key="field_selection_form"):
                    cols = st.columns(3)
                    current_selection_in_form = []
                    selected_in_state = st.session_state.get('selected_fields', [])
                    for i, field in enumerate(suggested_fields):
                        with cols[i % 3]:
                            # Checkbox inside the form
                            is_selected = st.checkbox(field, value=(field in selected_in_state), key=f"cb_{field}")
                            if is_selected:
                                current_selection_in_form.append(field)
                    
                    # Form submission button
                    submitted = st.form_submit_button("Confirm Fields & Start Extraction", type="primary")
                    
                    if submitted:
                        if not current_selection_in_form:
                            st.warning("Please select at least one field to extract.")
                        else:
                            print(f"Fields confirmed via form: {current_selection_in_form}. Proceeding.")
                            st.session_state.selected_fields = current_selection_in_form
                            # Clear downstream results
                            st.session_state.extraction_result = None
                            st.session_state.review_result = None
                            st.session_state.step = 4 # Move to extraction step
                            st.rerun()
            else:
                st.warning("Classification result not available. Please upload a document.")

    # Step 4 -> 5: Perform Extraction
    elif current_step == 4 and st.session_state.ocr_text and st.session_state.selected_fields:
        print("Step 4 -> 5: Extracting fields")
        try:
            with status_placeholder:
                 with st.spinner("✍️ Extracting fields..."):
                    extraction_res = extract_fields(
                        st.session_state.ocr_text,
                        st.session_state.selected_fields
                    )
            print(f"Extraction Result: {extraction_res}")
            st.session_state.extraction_result = extraction_res
            st.session_state.step = 5 # Move to display extraction
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during extraction: {e}")
            reset_app_state()

    # Step 5: Display Extraction Results & Transition to Pause/Review
    elif current_step == 5:
        print("Step 5: Displaying Extraction Results")
        with main_area_placeholder:
            st.subheader("Field Extraction Results")
            if st.session_state.extraction_result is not None:
                df_extract = pd.DataFrame(
                    list(st.session_state.extraction_result.items()),
                    columns=['Field', 'Extracted Value']
                )
                st.dataframe(df_extract, use_container_width=True, hide_index=True)
                
                # Check for potentially problematic extraction
                if not any(val for val in st.session_state.extraction_result.values() if val is not None and str(val).strip() not in ["", "null", "None"]):
                     st.warning("Extraction found no values or only nulls. Reviewing results...")
                
                # --- Auto-transition to Review Step after display ---
                # This block essentially finishes step 5 and immediately sets up for step 6
                print("Step 5 complete. Setting step to 6 to initiate pause and review.")
                st.session_state.step = 6
                # We need to rerun for the pause and step 6 logic to execute
                st.rerun() # Rerun to trigger the pause and step 6 logic
            else:
                 st.warning("Extraction data is missing. Cannot proceed to review.")
                 reset_app_state()

    # Step 6 -> 7: Pause and Perform Review
    elif current_step == 6:
        print("Step 6: Pausing before review...")
        # Display extraction results while pausing
        with main_area_placeholder:
            st.subheader("Field Extraction Results") # Re-display results
            if st.session_state.extraction_result is not None:
                 df_extract = pd.DataFrame(
                    list(st.session_state.extraction_result.items()),
                    columns=['Field', 'Extracted Value']
                 )
                 st.dataframe(df_extract, use_container_width=True, hide_index=True)
            else:
                 st.warning("Extraction data missing.")

        # THE PAUSE
        time.sleep(2)

        print("Step 6 -> 7: Performing Review")
        if st.session_state.ocr_text and st.session_state.extraction_result is not None:
            try:
                with status_placeholder: # Show spinner during review
                    with st.spinner("🧐 Reviewing extraction..."):
                        review_res = review_fields(
                            st.session_state.ocr_text,
                            st.session_state.extraction_result
                        )
                print(f"Review Result: {review_res}")
                st.session_state.review_result = review_res
                st.session_state.step = 7 # Review done, move to final display
                st.rerun()
            except Exception as e:
                 st.error(f"An error occurred during review: {e}")
                 reset_app_state()
        else:
            st.warning("Cannot perform review - missing text or extraction data.")
            reset_app_state() # Reset if prerequisites are missing

    # Step 7: Display Final Review Results
    elif current_step == 7:
        print("Step 7: Displaying Final Review Results")
        with main_area_placeholder:
            st.subheader("Review & Feedback Results")
            if st.session_state.review_result is not None and st.session_state.extraction_result is not None:
                display_data = []
                # Ensure all extracted fields are shown, even if missing from review
                for field in st.session_state.selected_fields:
                    extracted_value = st.session_state.extraction_result.get(field, "Error: Not Extracted")
                    review = st.session_state.review_result.get(field, {"status": "ERROR", "feedback": "Not in review result"})
                    status_icon = "✅" if review.get("status") == "PASS" else ("❌" if review.get("status") == "FAIL" else "❓")
                    display_data.append({
                        'Field': field,
                        'Extracted Value': extracted_value,
                        'Status': f'{status_icon} {review.get("status")}',
                        'Feedback': review.get("feedback", "")
                    })
                df_review = pd.DataFrame(display_data)
                st.dataframe(df_review, use_container_width=True, hide_index=True)

                st.divider()
                # Download button
                try:
                    # Combine extracted + review? For now, just extracted.
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
                 st.warning("Review or extraction results not available for final display.")

    # --- Footer/Debug Info (Optional) ---
    # st.divider()
    # with st.expander("Debug: Show Session State"):
    #    st.write(st.session_state)