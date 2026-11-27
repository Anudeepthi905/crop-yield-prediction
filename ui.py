"""Shared UI pieces for the Streamlit pages: copy, constants and components."""
import pandas as pd
import pytesseract
import streamlit as st

from core import predict_yields, read_npk_from_bytes

YIELD_UNIT_NOTE = "Yields are in the units of the source dataset, which doesn't specify them."

PAPER_URL = "https://doi.org/10.36287/setsci.17.1.0040"
REPO_URL = "https://github.com/Anudeepthi905/crop-yield-prediction"

ABOUT_NOTE = (
    "This app reads the **Nitrogen (N), Phosphorus (P) and Potassium (K)** values from a Soil Health Card "
    "(or takes them typed in), predicts the yield of Rice, Maize, Groundnut and Redgram, and suggests the crop "
    "with the highest predicted yield.\n\n"
    "It is an **academic project**: the models were trained on a small dataset, so predictions are "
    "**estimates only** and not agronomic advice."
)

INPUT_UPLOAD = "Upload a Soil Health Card"
INPUT_SAMPLE = "Try a sample card"
INPUT_MANUAL = "Enter values manually"


def algo(n, p, k):
    py = predict_yields(n, p, k)
    best_crop, best_yield = max(py.items(), key=lambda item: item[1][0])
    st.subheader("Crop Suggestion and Yield Prediction")
    st.success(f"Suggested crop: **{best_crop}**, estimated yield **{best_yield[0]:.2f}**")
    with st.expander("Estimated yield for every crop"):
        table = pd.DataFrame(
            {"Crop": list(py), "Estimated yield": [round(float(v[0]), 2) for v in py.values()]})
        st.dataframe(table.sort_values(table.columns[1], ascending=False), hide_index=True)
    st.caption(f"{YIELD_UNIT_NOTE} Estimates from models trained on a small academic dataset. "
               "Not agronomic advice.")


def fill_values_from_image(image_bytes, source_id):
    """OCR the image, show it, and pre-fill the N/P/K inputs (once per new image)."""
    try:
        with st.spinner("Reading values from the card..."):
            found = read_npk_from_bytes(image_bytes)
    except pytesseract.TesseractNotFoundError:
        st.error("The OCR engine (Tesseract) isn't available here, so values can't be read from images. "
                 "Please type them below.")
        return
    if found is None:
        st.error("This file couldn't be opened as an image. Please upload a JPG or PNG photo or screenshot "
                 "of the card, or type the values below.")
        return
    st.image(image_bytes, width=520)

    if st.session_state.get("filled_from") != source_id:
        for key, value in found.items():
            st.session_state[f"npk_{key}"] = value
        st.session_state["filled_from"] = source_id

    names = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)"}
    missing = [names[key] for key in names if key not in found]
    if not missing:
        st.success("Values read from the card. Please check them below and correct any mistakes.")
    elif len(missing) < 3:
        st.warning(f"Couldn't read {', '.join(missing)} from this image. Please type the missing values below.")
    else:
        st.warning("Couldn't read the values from this image. Try a clearer, straight-on photo of the "
                   "\"Soil Test Results\" table, or type the values below.")
