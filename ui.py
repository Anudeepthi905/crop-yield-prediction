"""Shared UI pieces for the Streamlit pages: copy, constants and components."""
import pandas as pd
import pytesseract
import streamlit as st

from core import CROP_MODELS, FEATURES, PLAUSIBLE_RANGE, load_training_data, predict_yields, read_npk_from_bytes

NUTRIENTS = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}
FIELD_KEY = {key: f"npk_{key}" for key in NUTRIENTS}
# Hard limits only catch typos (same bounds as the OCR plausibility check). Real cards often sit outside
# the training data, so that gets a soft warning instead of a block.
INPUT_MAX = {key: float(high) for key, (_, high) in PLAUSIBLE_RANGE.items()}

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


@st.cache_data(show_spinner=False)
def training_ranges():
    """Min and max of N, P and K across the training data: {"N": (low, high), ...}."""
    data = pd.concat([load_training_data(csv_name) for csv_name, *_ in CROP_MODELS.values()])
    return {key: (float(data[key].min()), float(data[key].max())) for key in FEATURES}


@st.cache_data(show_spinner=False, max_entries=500)
def predict_cached(n, p, k):
    """Predicted yield per crop, highest first. Cached, so repeating the same inputs is instant."""
    yields = {crop: float(value[0]) for crop, value in predict_yields(n, p, k).items()}
    return sorted(yields.items(), key=lambda item: item[1], reverse=True)


def init_state():
    for key in FIELD_KEY.values():
        st.session_state.setdefault(key, 0.0)


def current_inputs():
    return tuple(float(st.session_state[FIELD_KEY[key]]) for key in NUTRIENTS)


def outside_training(values):
    """Readable notes for each value outside the training data's range."""
    notes = []
    for (key, (low, high)), value in zip(training_ranges().items(), values):
        if not low <= value <= high:
            notes.append(f"{key} = {value:g} (trained on {low:g}–{high:g})")
    return notes


def npk_form():
    """The N, P, K form. One rerun per submit; Enter submits. Stores the result in session_state."""
    ranges = training_ranges()
    with st.form("npk_form"):
        columns = st.columns(3)
        for column, (key, name) in zip(columns, NUTRIENTS.items()):
            low, high = ranges[key]
            column.number_input(
                f"{name} ({key}), kg/ha", min_value=0.0, max_value=INPUT_MAX[key], step=1.0, format="%.2f",
                key=FIELD_KEY[key],
                help=f"\"Available {name}\" on your Soil Health Card. The models were trained on "
                     f"{low:g}–{high:g} kg/ha; values outside that still work but are less reliable.")
        if st.form_submit_button("Predict", type="primary", icon=":material/eco:"):
            _predict(current_inputs())


def _predict(values):
    if all(value == 0 for value in values):
        st.error("**Enter your soil test values first.** N, P and K are all 0, which isn't a real soil "
                 "test, so there's nothing to predict from.", icon=":material/error:")
        return
    try:
        with st.spinner("Predicting..."):
            ranking = predict_cached(*values)
    except Exception:  # noqa: BLE001 - show a friendly message instead of a traceback
        st.error("**We couldn't make a prediction just now.** Please try again. If it keeps happening, "
                 "reload the page.", icon=":material/error:")
        return
    st.session_state["result"] = {"inputs": values, "ranking": ranking}


def show_result(result):
    (best_crop, best_yield), *_ = result["ranking"]
    st.subheader("Crop Suggestion and Yield Prediction")
    st.success(f"Suggested crop: **{best_crop}**, estimated yield **{best_yield:.2f}**")
    with st.expander("Estimated yield for every crop"):
        st.dataframe(pd.DataFrame(result["ranking"], columns=["Crop", "Estimated yield"]).round(2),
                     hide_index=True)
    for note in outside_training(result["inputs"]):
        st.caption(f"Outside the training data: {note}. Treat the prediction as a rough guide.")
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
