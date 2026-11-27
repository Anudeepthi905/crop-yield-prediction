"""Predict page: get N, P, K values and show the recommendation."""
import streamlit as st

from core import ALLOWED_TYPES, MAX_UPLOAD_MB, SAMPLES, SAMPLES_DIR
from ui import ABOUT_NOTE, INPUT_MANUAL, INPUT_SAMPLE, INPUT_UPLOAD, algo, fill_values_from_image

st.title("Crop Yield Predictor")
st.info(ABOUT_NOTE)

for key in ("npk_N", "npk_P", "npk_K"):
    st.session_state.setdefault(key, 0.0)

st.subheader("1. Provide your soil's N, P and K values")
option = st.radio("Input method", (INPUT_UPLOAD, INPUT_SAMPLE, INPUT_MANUAL), horizontal=True)

if option == INPUT_UPLOAD:
    uploaded_file = st.file_uploader(
        "Photo or screenshot of the card's \"Soil Test Results\" table",
        type=ALLOWED_TYPES, help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB.")
    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is {uploaded_file.size / 2**20:.1f} MB. Please upload an image under "
                     f"{MAX_UPLOAD_MB} MB.")
        else:
            image_bytes = uploaded_file.getvalue()
            fill_values_from_image(image_bytes, f"upload:{uploaded_file.name}:{len(image_bytes)}")

elif option == INPUT_SAMPLE:
    st.caption("No card handy? Pick a sample: a real card (full or cropped) or one of three synthetic ones.")
    choice = st.selectbox("Sample card", list(SAMPLES))
    if st.button("Try a sample"):
        st.session_state["active_sample"] = choice
        st.session_state["filled_from"] = None  # re-fill the values even if this sample was used before
    active = st.session_state.get("active_sample")
    if active == choice:
        fill_values_from_image((SAMPLES_DIR / SAMPLES[active]).read_bytes(), f"sample:{active}")

st.subheader("2. Check the values (kg/ha)")
col_n, col_p, col_k = st.columns(3)
n = col_n.number_input("Nitrogen (N)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_N")
p = col_p.number_input("Phosphorus (P)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_P")
k = col_k.number_input("Potassium (K)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_K")

st.subheader("3. Predict")
if st.button("Predict", type="primary"):
    if n == 0 and p == 0 and k == 0:
        st.warning("Please enter the N, P and K values first.")
    else:
        algo(n, p, k)
