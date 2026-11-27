"""Predict page: get N, P, K values and show the recommendation."""
import streamlit as st

import ui
from core import ALLOWED_TYPES, MAX_UPLOAD_MB, SAMPLES, SAMPLES_DIR
from ui import ABOUT_NOTE, INPUT_MANUAL, INPUT_MODES, INPUT_SAMPLE, INPUT_UPLOAD

st.title("Crop Yield Predictor")
st.info(ABOUT_NOTE)

ui.init_state()

st.subheader("1. Provide your soil's N, P and K values")
mode = st.radio("How do you want to add your soil values?", INPUT_MODES, key="input_mode", horizontal=True)

if mode == INPUT_UPLOAD:
    uploaded_file = st.file_uploader(
        "Soil Health Card photo or screenshot", type=ALLOWED_TYPES,
        help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB. A straight-on photo of the \"Soil Test Results\" table "
             "works best.")
    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"**That file is {uploaded_file.size / 2**20:.1f} MB.** Upload an image under "
                     f"{MAX_UPLOAD_MB} MB, for example a screenshot of the card.", icon=":material/error:")
        else:
            ui.read_card(uploaded_file.getvalue(), f"upload:{uploaded_file.file_id}", uploaded_file.name)

elif mode == INPUT_SAMPLE:
    choice = st.selectbox("Sample card", list(SAMPLES),
                          help="One real card (full and cropped) and three synthetic ones.")
    if st.button("Load sample", icon=":material/upload_file:"):
        st.session_state["active_sample"] = choice
        st.session_state["ocr"] = None  # read again, so the form is re-filled even for the same sample
    if st.session_state.get("active_sample") == choice:
        ui.read_card((SAMPLES_DIR / SAMPLES[choice]).read_bytes(), f"sample:{choice}", choice)

elif mode == INPUT_MANUAL:
    st.caption("Type the values from the \"Soil Test Results\" table on your Soil Health Card.")

st.subheader("2. Check the values")
ui.npk_form()

st.subheader("3. Recommendation")
ui.results_section()
