"""Shared UI pieces for the Streamlit pages: copy, constants and components."""
import altair as alt
import pandas as pd
import pytesseract
import streamlit as st

from core import (CROP_MODELS, FEATURES, OCR_VERSION, PLAUSIBLE_RANGE, load_training_data, predict_yields,
                  read_npk_from_bytes)

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
            notes.append(f"{key} {value:g} (trained on {low:g}–{high:g})")
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
    st.session_state["show_result"] = True   # scroll to / focus the new result once (see _bring_into_view)


def fmt_yield(value):
    return f"{value:.2f} {YIELD_UNIT}" if YIELD_UNIT else f"{value:.2f}"


def results_section():
    """Empty state before the first prediction, then the result card (kept across reruns)."""
    result = st.session_state.get("result")
    if not result:
        with st.container(border=True):
            st.markdown("**Your recommendation will appear here**")
            st.caption("Add your soil values and press **Predict**. You'll see the best crop for your "
                       "soil, its estimated yield, and how all four crops compare.")
        return
    stale = current_inputs() != result["inputs"]
    if stale:
        st.info("The values have changed since this result. Press **Predict** to update it.",
                icon=":material/update:")
    _result_card(result, stale)
    if st.session_state.pop("show_result", False):
        _bring_into_view()


def _result_card(result, stale=False):
    ranking = result["ranking"]
    (best, best_yield), (runner_up, runner_up_yield) = ranking[0], ranking[1]
    n, p, k = result["inputs"]
    # A result for old inputs is faded (CSS on the key) so it can't be mistaken for the current one.
    with st.container(border=True, key="result_card_stale" if stale else "result_card"):
        best_col, yield_col, lead_col = st.columns(3)
        best_col.metric("Recommended crop", best)
        yield_col.metric("Estimated yield", fmt_yield(best_yield))
        lead_col.metric(f"Ahead of {runner_up}", f"+{fmt_yield(best_yield - runner_up_yield)}",
                        help=f"{runner_up} is next, at {fmt_yield(runner_up_yield)}.")
        st.altair_chart(_ranking_chart(ranking), width="stretch",
                        alt="Estimated yield for each crop: "
                            + ", ".join(f"{crop} {fmt_yield(value)}" for crop, value in ranking))
        st.caption(f"Based on N {n:.2f}, P {p:.2f} and K {k:.2f} kg/ha. {YIELD_UNIT_NOTE}")
        notes = outside_training(result["inputs"])
        if notes:
            st.caption(f":material/info: Beyond the training data for {', '.join(notes)}, so this estimate is "
                       "less certain.")
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


def _bring_into_view():
    """After Predict: on one-column layouts, scroll the result card into view; always move focus to it.

    Without this, on a phone the result appears below the fold and people press Predict again.
    The script is static (no user input in it); it runs only on the rerun right after a prediction.
    """
    st.html("""<script>
(() => {
  let tries = 0;
  const oneColumn = () => window.innerWidth < 1024;
  const go = () => {
    const card = document.querySelector('.st-key-result_card');
    if (!card) { if (tries++ < 120) setTimeout(go, 50); return; }      // wait up to ~6 s for the card
    card.setAttribute('tabindex', '-1');
    card.setAttribute('role', 'region');
    card.setAttribute('aria-label', 'Recommendation');
    if (oneColumn()) card.scrollIntoView({behavior: 'smooth', block: 'start'});
    card.focus({preventScroll: true});
    // Content can still be laying out (the chart renders a moment later); check again and correct.
    [700, 1500].forEach((delay) => setTimeout(() => {
      const top = card.getBoundingClientRect().top;
      if (oneColumn() && (top < -8 || top > window.innerHeight * 0.4)) card.scrollIntoView({block: 'start'});
    }, delay));
  };
  go();
})();
</script>""", unsafe_allow_javascript=True)


# ---------------------------------------------------------------------------
# Responsive and accessibility styles
# ---------------------------------------------------------------------------
# Streamlit doesn't expose these controls, so they're set here, in one place. Selectors use only
# Streamlit's data-testid attributes and the .st-key-<key> classes of containers this app names
# (checked against Streamlit 1.67; tests/e2e/test_responsive.py catches regressions).
RESPONSIVE_CSS = """<style>
/* Headings scale with the screen and never break mid-word ("recommendatio / n" at 360 px). */
h1 { font-size: clamp(1.85rem, 4.5vw + 0.9rem, 2.75rem) !important; line-height: 1.15 !important;
     word-break: normal !important; overflow-wrap: normal !important; }
h2, h3 { word-break: normal !important; overflow-wrap: normal !important; }

/* 16 px inputs: iPhone Safari zooms into any field with smaller text when it's tapped. */
[data-testid="stNumberInputField"] { font-size: 16px !important; }

/* Captions: darker grey (about 8:1 on white) so they stay readable in sunlight. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #4B5148 !important; opacity: 1 !important; }

/* Touch screens: tap targets of at least 44 x 44 px. */
@media (pointer: coarse), (max-width: 640px) {
  button[data-testid^="stBaseButton"] { min-height: 44px; }
  [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { min-width: 44px; min-height: 44px; }
  [data-testid="stNumberInputContainer"], [data-testid="stNumberInputField"] { min-height: 44px; }
  [data-testid="stTooltipHoverTarget"] { min-width: 44px; min-height: 44px;
     display: inline-flex !important; align-items: center; justify-content: center; }
  [data-testid="stSidebarNavLink"], [data-testid="stRadioOption"] { min-height: 44px; align-items: center; }
  [data-testid="stExpandSidebarButton"], [data-testid="stHeader"] button { min-width: 44px; min-height: 44px; }
}

/* Phones: input-method options stacked instead of wrapping unevenly. */
@media (max-width: 640px) {
  [data-testid="stRadioGroup"] { flex-direction: column !important; align-items: flex-start !important; gap: 0 !important; }
}

/* Predict page: inputs and result side by side from 1024 px, result kept in view while scrolling. */
[data-testid="stMainBlockContainer"]:has(.st-key-predict_layout) { max-width: 1240px; }
@media (min-width: 1024px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-predict_result) > [data-testid="stColumn"]:has(.st-key-predict_result) {
    position: sticky; top: 4.5rem; align-self: flex-start; }
}
@media (max-width: 1023.98px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-predict_result) { flex-direction: column !important; }
  [data-testid="stHorizontalBlock"]:has(.st-key-predict_result) > [data-testid="stColumn"] {
    width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important; }
}

/* Phones: crop and yield side by side, the lead on its own line below, so the result stays compact
   and the chart stays on screen (instead of three tall stacked blocks). Labels wrap, never truncate. */
@media (max-width: 640px) {
  :is(.st-key-result_card, .st-key-result_card_stale) [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important; gap: 0.25rem 0.75rem !important; }
  :is(.st-key-result_card, .st-key-result_card_stale) [data-testid="stColumn"] {
    min-width: calc(50% - 0.375rem) !important; width: calc(50% - 0.375rem) !important;
    flex: 1 1 calc(50% - 0.375rem) !important; }
  :is(.st-key-result_card, .st-key-result_card_stale) [data-testid="stColumn"]:last-child {
    min-width: 100% !important; width: 100% !important; flex-basis: 100% !important; }
  :is(.st-key-result_card, .st-key-result_card_stale) [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
  :is(.st-key-result_card, .st-key-result_card_stale) [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
}

/* A result for old inputs is faded until Predict is pressed again. */
.st-key-result_card_stale { opacity: 0.45; }

/* Home page: crop photos in a 2 x 2 grid on phones instead of four full-width photos. */
@media (max-width: 640px) {
  .st-key-crop_grid [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.75rem !important; }
  .st-key-crop_grid [data-testid="stColumn"] {
    min-width: calc(50% - 0.375rem) !important; width: calc(50% - 0.375rem) !important;
    flex: 1 1 calc(50% - 0.375rem) !important; }
}
</style>"""


# Phone menu. Below 768 px Streamlit folds the top navigation into a sidebar drawer opened by an
# unlabelled "»" button, and the drawer stays open over the page after a link is chosen.
MENU_CSS = """<style>
/* The drawer's open button becomes a labelled hamburger: three bars and "Menu". */
[data-testid="stExpandSidebarButton"] {
  width: auto !important; min-height: 44px; padding: 0 0.85rem 0 0.75rem !important; gap: 0.6rem;
  border-radius: 0.5rem; color: #1B1B1B !important; }
[data-testid="stExpandSidebarButton"] > span { display: none !important; }
[data-testid="stExpandSidebarButton"]::before {
  content: ""; width: 20px; height: 2px; border-radius: 1px; background: currentColor;
  box-shadow: 0 -6px 0 currentColor, 0 6px 0 currentColor; }
[data-testid="stExpandSidebarButton"]::after { content: "Menu"; font-size: 1rem; font-weight: 600; }
/* The drawer's close button becomes an X. */
[data-testid="stSidebarCollapseButton"] button { min-width: 44px; min-height: 44px; color: #1B1B1B !important; }
[data-testid="stSidebarCollapseButton"] button > span { display: none !important; }
[data-testid="stSidebarCollapseButton"] button::before { content: "\\2715"; font-size: 1.25rem; line-height: 1; }
/* While the drawer is open (its "Menu" button is gone), dim the page behind it. */
@media (max-width: 767.98px) {
  [data-testid="stApp"]:not(:has([data-testid="stExpandSidebarButton"])) [data-testid="stSidebar"] {
    box-shadow: 0 0 0 100vmax rgba(20, 26, 20, 0.4); }
}
/* Menu links: full-width rows with room to tap. */
[data-testid="stSidebarNavLink"] { min-height: 48px; padding-left: 0.75rem; }
[data-testid="stSidebarNavLink"] span { font-size: 1.05rem; }
</style>"""

# Closes the drawer after a link is chosen, on a tap outside it, or on Escape (narrow screens only).
# Static script, installed once per browser tab.
MENU_JS = """<script>
(() => {
  if (window.__cropMenuInstalled) return;
  window.__cropMenuInstalled = true;
  const sidebar = () => document.querySelector('[data-testid="stSidebar"]');
  const isOpen = () => {
    const el = sidebar();
    if (!el || window.innerWidth >= 768) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.right > 1;
  };
  const close = () => {
    const btn = document.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (btn && isOpen()) btn.click();
  };
  document.addEventListener('click', (event) => {
    if (!isOpen()) return;
    if (event.target.closest('[data-testid="stSidebarNavLink"]')) { setTimeout(close, 60); return; }
    if (!event.target.closest('[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"]')) close();
  }, true);
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });
})();
</script>"""


def apply_styles():
    st.html(RESPONSIVE_CSS + MENU_CSS)
    st.html(MENU_JS, unsafe_allow_javascript=True)


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
        result = read_npk_from_bytes(image_bytes, OCR_VERSION)
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
