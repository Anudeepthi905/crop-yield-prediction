"""Smoke tests. Run from the project root with:  pytest"""
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import ui
import core as app

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = str(ROOT / "app.py")
PAGE_FILES = {"Home": "views/home.py", "Predict": "views/predict.py", "About": "views/about.py"}

# Expected N, P, K for each sample card in samples/
EXPECTED_SAMPLE_NPK = {
    "sample_card_real_full.jpg": (305.0, 36.0, 69.0),
    "sample_card_synthetic_full.png": (240.0, 45.0, 210.0),
    "sample_card_real_cropped.jpg": (305.0, 36.0, 69.0),
    "sample_card_synthetic_maize.png": (160.0, 20.0, 150.0),
    "sample_card_synthetic_rice.png": (200.0, 8.0, 280.0),
}


def tesseract_available():
    app.configure_tesseract()
    return shutil.which(app.pytesseract.pytesseract.tesseract_cmd) is not None


needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="Tesseract OCR not installed")


def load_image(path):
    # np.fromfile + imdecode handles paths with spaces/non-ASCII on Windows (cv2.imread does not).
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"{path} could not be decoded"
    return img


def new_app():
    at = AppTest.from_file(APP_FILE, default_timeout=120)
    at.run()
    return at


# --- OCR -------------------------------------------------------------------

def test_every_sample_has_an_expected_value():
    assert set(app.SAMPLES.values()) == set(EXPECTED_SAMPLE_NPK)


@needs_tesseract
@pytest.mark.parametrize("filename,expected", EXPECTED_SAMPLE_NPK.items())
def test_ocr_reads_sample_cards(filename, expected):
    assert app.extract_npk_from_image(load_image(app.SAMPLES_DIR / filename)) == expected


@needs_tesseract
@pytest.mark.parametrize("filename,expected", EXPECTED_SAMPLE_NPK.items())
def test_ocr_reads_enlarged_photos(filename, expected):
    """Phone photos come at other resolutions; a larger image must read the same values."""
    img = cv2.resize(load_image(app.SAMPLES_DIR / filename), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    assert app.extract_npk_from_image(img) == expected


@needs_tesseract
def test_ocr_small_screenshot_never_guesses_wrong():
    """A shrunken card may lose a value, but must never return a wrong one (e.g. 120 from '120-280')."""
    img = load_image(app.SAMPLES_DIR / "sample_card_real_cropped.jpg")
    small = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    found = app.read_npk(small)
    assert all(found[key] == value for key, value in {"N": 305.0, "P": 36.0, "K": 69.0}.items() if key in found)
    assert len(found) >= 2


@pytest.mark.parametrize("text,expected", [
    ("4 [Available Nitrogen (N) | 305.00 kg/ha", {"N": 305.0}),
    ("5 Available Phosphorus (P)| 36 00;kgiha High", {"P": 36.0}),   # dot read as a space
    ("6 Available Potassium (K) | 69,00 kg/ha", {"K": 69.0}),         # dot read as a comma
    ("5 Available Phosphorus (P) | 3600 kg/ha", {}),                  # implausible: rejected
    ("6 Available Potassium (K) | 69.00ikg/ha 120-280kgha", {"K": 69.0}),
    ("6 Available Potassium (K) | ikgha Low 120-280kg/ha", {}),        # value unreadable: never take the range
    ("Soil Test Results", {}),
])
def test_parse_npk(text, expected):
    assert app.parse_npk(text) == expected


def test_unreadable_bytes_return_none():
    assert app.read_npk_from_bytes(b"this is not an image") is None


# --- Models ------------------------------------------------------------------

def test_predict_yields_returns_all_crops():
    yields = app.predict_yields(305.0, 36.0, 69.0)
    assert set(yields) == {"Redgram", "Rice", "Maize", "Groundnut"}
    for value in yields.values():
        assert math.isfinite(float(value[0]))


def test_all_referenced_files_exist():
    source = (ROOT / "views" / "home.py").read_text(encoding="utf-8")
    images = re.findall(r'"(\w+\.jpg)"', source)
    assert images, "expected app.py to reference images"
    for name in images:
        assert (app.IMAGES_DIR / name).is_file(), f"missing image: Images/{name}"
    for csv_name, *_ in app.CROP_MODELS.values():
        assert (app.DATA_DIR / csv_name).is_file(), f"missing data file: data/{csv_name}"
    for filename in app.SAMPLES.values():
        assert (app.SAMPLES_DIR / filename).is_file(), f"missing sample: samples/{filename}"


# --- UI ------------------------------------------------------------------------

@pytest.mark.parametrize("page", list(PAGE_FILES))
def test_pages_render_without_errors(page):
    at = new_app()
    at.switch_page(PAGE_FILES[page]).run()
    assert not at.exception


def predict_page():
    at = new_app()
    at.switch_page(PAGE_FILES["Predict"]).run()
    assert not at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception


def enter_values(at, n, p, k):
    at.number_input(key="npk_N").set_value(n)
    at.number_input(key="npk_P").set_value(p)
    at.number_input(key="npk_K").set_value(k)


def metrics(at):
    return {m.label: m.value for m in at.metric}


def load_sample(at, name):
    at.radio(key="input_mode").set_value(ui.INPUT_SAMPLE).run()
    next(s for s in at.selectbox if s.label == "Sample card").select(name).run()
    click(at, "Load sample")


def test_empty_state_before_first_prediction():
    at = predict_page()
    assert any("Your recommendation will appear here" in m.value for m in at.markdown)
    assert not at.metric


def test_all_zero_input_is_blocked():
    at = predict_page()
    click(at, "Predict")
    assert "Enter your soil test values first" in at.error[0].value
    assert "result" not in at.session_state
    assert not at.metric


def test_valid_prediction_shows_results_card():
    at = predict_page()
    at.radio(key="input_mode").set_value(ui.INPUT_MANUAL).run()
    enter_values(at, 305.0, 36.0, 69.0)
    click(at, "Predict")
    shown = metrics(at)
    assert shown["Recommended crop"] == "Maize"
    assert re.fullmatch(r"\d+\.\d{2}", shown["Estimated yield"])  # 2 decimals, no 15-digit float
    assert any(label.startswith("Ahead of ") for label in shown)
    assert any("Based on N 305.00, P 36.00 and K 69.00" in c.value for c in at.caption)
    # Outside the training data (N > 277, K < 78): flagged, not blocked
    assert any("Beyond the training data for N 305" in c.value for c in at.caption)


def test_result_survives_reruns_and_flags_changed_inputs(monkeypatch):
    monkeypatch.setattr(ui, "read_npk_from_bytes", lambda image_bytes: {"N": 200.0, "P": 8.0, "K": 280.0})
    at = predict_page()
    at.radio(key="input_mode").set_value(ui.INPUT_MANUAL).run()
    enter_values(at, 160.0, 20.0, 150.0)
    click(at, "Predict")
    assert metrics(at)["Recommended crop"] == "Maize"
    load_sample(at, "Synthetic card B")   # changes the form values, not the result
    assert metrics(at)["Recommended crop"] == "Maize"
    assert any("changed since this result" in i.value for i in at.info)


def test_prediction_failure_shows_friendly_error(monkeypatch):
    def broken(*values):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(ui, "predict_cached", broken)
    at = predict_page()
    enter_values(at, 200.0, 30.0, 200.0)
    click(at, "Predict")
    assert "couldn't make a prediction" in at.error[0].value
    assert "model exploded" not in at.error[0].value   # no internals leak to the visitor
    assert not at.metric


def test_ocr_failure_shows_recovery_button(monkeypatch):
    monkeypatch.setattr(ui, "read_npk_from_bytes", lambda image_bytes: {})
    at = predict_page()
    load_sample(at, "Synthetic card A")
    assert "couldn't find N, P and K" in at.error[0].value
    click(at, "Type the values instead")
    assert at.radio(key="input_mode").value == ui.INPUT_MANUAL


def test_card_is_read_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "read_npk_from_bytes",
                        lambda image_bytes: calls.append(1) or {"N": 200.0, "P": 8.0, "K": 280.0})
    at = predict_page()
    load_sample(at, "Synthetic card B")
    click(at, "Predict")                  # a rerun with the same card must not read it again
    assert len(calls) == 1
    assert metrics(at)["Recommended crop"] == "Rice"


@needs_tesseract
def test_try_a_sample_fills_values_and_predicts():
    at = predict_page()
    load_sample(at, "Synthetic card B")
    assert (at.number_input(key="npk_N").value, at.number_input(key="npk_P").value,
            at.number_input(key="npk_K").value) == (200.0, 8.0, 280.0)
    # The user can correct a value before predicting
    at.number_input(key="npk_P").set_value(18.0)
    click(at, "Predict")
    assert metrics(at)["Recommended crop"] == "Rice"
