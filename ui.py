"""Shared UI pieces for the Streamlit pages: copy, constants and components."""
import altair as alt
import pandas as pd
import pytesseract
import streamlit as st

from core import CROP_MODELS, FEATURES, PLAUSIBLE_RANGE, load_training_data, predict_yields, read_npk_from_bytes

NUTRIENTS = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}
FIELD_KEY = {key: f"npk_{key}" for key in NUTRIENTS}
# Hard limits only catch typos (same bounds as the OCR plausibility check). Real cards often sit outside
# the training data, so that gets a soft warning instead of a block.
INPUT_MAX = {key: float(high) for key, (_, high) in PLAUSIBLE_RANGE.items()}

# The training data doesn't state the yield unit, so results are plain numbers. Set this one constant
# (e.g. "q/ha") if the unit is confirmed, and every result and the chart will show it.
YIELD_UNIT = ""
YIELD_UNIT_NOTE = "The dataset doesn't state the yield unit."

# Same colour as primaryColor in .streamlit/config.toml: used only for primary buttons and the winning result.
BRAND_COLOR = "#2E7D32"
MUTED_COLOR = "#C9CFC6"

PAPER_URL = "https://doi.org/10.36287/setsci.17.1.0040"
REPO_URL = "https://github.com/Anudeepthi905/crop-yield-prediction"

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


def fmt_yield(value):
    return f"{value:.2f} {YIELD_UNIT}" if YIELD_UNIT else f"{value:.2f}"


def results_section():
    """Empty state before the first prediction, then the result card (kept across reruns)."""
    result = st.session_state.get("result")
    if not result:
        with st.container(border=True):
            st.markdown("**Your recommendation will appear here**")
            st.caption("Add your soil values above and press **Predict**. You'll see the best crop for your "
                       "soil, its estimated yield, and how all four crops compare.")
        return
    if current_inputs() != result["inputs"]:
        st.info("The values above have changed since this result. Press **Predict** to update it.",
                icon=":material/update:")
    _result_card(result)


def _result_card(result):
    ranking = result["ranking"]
    (best, best_yield), (runner_up, runner_up_yield) = ranking[0], ranking[1]
    n, p, k = result["inputs"]
    with st.container(border=True):
        st.badge("Recommended crop", icon=":material/eco:", color="primary")
        best_col, yield_col, lead_col = st.columns(3)
        best_col.metric("Crop", best)
        yield_col.metric("Estimated yield", fmt_yield(best_yield))
        lead_col.metric(f"Ahead of {runner_up}", f"+{fmt_yield(best_yield - runner_up_yield)}",
                        help=f"{runner_up} is next, at {fmt_yield(runner_up_yield)}.")
        st.altair_chart(_ranking_chart(ranking), width="stretch",
                        alt="Estimated yield for each crop: "
                            + ", ".join(f"{crop} {fmt_yield(value)}" for crop, value in ranking))
        st.caption(f"Based on N {n:.2f}, P {p:.2f} and K {k:.2f} kg/ha. {YIELD_UNIT_NOTE}")
        for note in outside_training(result["inputs"]):
            st.caption(f":material/warning: Outside the training data: {note}. Treat this as a rough guide.")
        with st.expander("Exact values"):
            table = pd.DataFrame(ranking, columns=["Crop", "Estimated yield"])
            table.insert(0, "Rank", range(1, len(table) + 1))
            st.dataframe(table, hide_index=True, column_config={
                "Estimated yield": st.column_config.NumberColumn(format="%.2f")})


def _ranking_chart(ranking):
    """Horizontal bars, best first. The winner is in the brand colour AND labelled, so colour isn't the only cue."""
    data = pd.DataFrame(ranking, columns=["Crop", "Yield"])
    data["Label"] = [f"{crop} (recommended)" if i == 0 else crop for i, crop in enumerate(data["Crop"])]
    data["Best"] = [i == 0 for i in range(len(data))]
    data["Value"] = data["Yield"].map(fmt_yield)
    base = alt.Chart(data).encode(
        y=alt.Y("Label:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=13, ticks=False, domain=False)),
        x=alt.X("Yield:Q", title=None, axis=None))
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.condition("datum.Best", alt.value(BRAND_COLOR), alt.value(MUTED_COLOR)),
        tooltip=[alt.Tooltip("Crop:N"), alt.Tooltip("Value:N", title="Estimated yield")])
    labels = base.mark_text(align="left", dx=6, fontSize=13).encode(text="Value:N")
    return (bars + labels).properties(height=44 * len(data))


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
