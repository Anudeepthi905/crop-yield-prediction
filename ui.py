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

INPUT_UPLOAD = "Upload a card"
INPUT_SAMPLE = "Try a sample"
INPUT_MANUAL = "Type the values"
INPUT_MODES = (INPUT_UPLOAD, INPUT_SAMPLE, INPUT_MANUAL)

# What went wrong while reading a card, and how to fix it.
OCR_PROBLEMS = {
    "not_image": ("This file isn't a readable image.",
                  "Upload a JPG or PNG photo or screenshot of the card."),
    "no_values": ("We couldn't find N, P and K on this card.",
                  "A sharp, straight-on photo of the \"Soil Test Results\" table, without glare, works best."),
    "no_ocr": ("Card reading isn't available right now.",
               "You can still type the values from the card."),
    "failed": ("Something went wrong while reading the card.",
               "Try another photo, or type the values in."),
}


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


def switch_to_manual():
    st.session_state["input_mode"] = INPUT_MANUAL


def read_card(image_bytes, source_id, caption):
    """Read N, P, K from a card image once per source, pre-fill the form, and show the outcome."""
    record = st.session_state.get("ocr")
    if not record or record["source"] != source_id:
        with st.status("Reading the card...", expanded=False) as status:
            record = _run_ocr(image_bytes, source_id)
            st.session_state["ocr"] = record
            for key, value in record["found"].items():
                st.session_state[FIELD_KEY[key]] = value
            if record["problem"]:
                status.update(label="Couldn't read the card", state="error")
            else:
                status.update(label="Card read", state="complete")
    _show_card_outcome(image_bytes, record, caption)


def _run_ocr(image_bytes, source_id):
    found, problem = {}, None
    try:
        result = read_npk_from_bytes(image_bytes)
    except pytesseract.TesseractNotFoundError:
        problem = "no_ocr"
    except Exception:  # noqa: BLE001 - any OCR failure becomes a friendly message
        problem = "failed"
    else:
        if result is None:
            problem = "not_image"
        elif not result:
            problem = "no_values"
        else:
            found = result
    return {"source": source_id, "found": found, "problem": problem}


def _show_card_outcome(image_bytes, record, caption):
    if record["problem"] == "not_image":
        message_area = st.container()
    else:
        thumbnail, message_area = st.columns([1, 2], vertical_alignment="center")
        thumbnail.image(image_bytes, caption=caption, width="stretch")
    with message_area:
        found = record["found"]
        missing = [key for key in NUTRIENTS if key not in found]
        if record["problem"]:
            title, fix = OCR_PROBLEMS[record["problem"]]
            st.error(f"**{title}** {fix}", icon=":material/error:")
            st.button("Type the values instead", icon=":material/keyboard:", on_click=switch_to_manual)
        elif missing:
            st.warning(f"**Read {', '.join(found)}, but not {', '.join(missing)}.** Type the missing "
                       f"value{'s' if len(missing) > 1 else ''} in the form below.", icon=":material/warning:")
        else:
            st.success("**Read N, P and K from the card.** Check them in the form below, then press Predict.",
                       icon=":material/check_circle:")
