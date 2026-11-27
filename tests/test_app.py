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


@pytest.mark.parametrize("text,expected", [
    ("4 [Available Nitrogen (N) | 305.00 kg/ha", {"N": 305.0}),
    ("5 Available Phosphorus (P)| 36 00;kgiha High", {"P": 36.0}),   # dot read as a space
    ("6 Available Potassium (K) | 69,00 kg/ha", {"K": 69.0}),         # dot read as a comma
    ("5 Available Phosphorus (P) | 3600 kg/ha", {}),                  # implausible: rejected
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
    images = re.findall(r'IMAGES_DIR / "([^"]+)"', source)
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


def test_manual_input_prediction():
    at = new_app()
    at.switch_page(PAGE_FILES["Predict"]).run()
    at.radio[0].set_value(ui.INPUT_MANUAL).run()
    at.number_input(key="npk_N").set_value(305.0)
    at.number_input(key="npk_P").set_value(36.0)
    at.number_input(key="npk_K").set_value(69.0)
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    assert "Suggested crop: **Maize**" in at.success[0].value


def test_predict_without_values_shows_warning():
    at = new_app()
    at.switch_page(PAGE_FILES["Predict"]).run()
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    assert "Enter your soil test values first" in at.error[0].value
    assert "result" not in at.session_state


@needs_tesseract
def test_try_a_sample_fills_values_and_predicts():
    at = new_app()
    at.switch_page(PAGE_FILES["Predict"]).run()
    at.radio[0].set_value(ui.INPUT_SAMPLE).run()
    sample_select = next(s for s in at.selectbox if s.label == "Sample card")
    sample_select.select("Synthetic card B").run()
    next(b for b in at.button if b.label == "Try a sample").click().run()
    assert not at.exception
    assert (at.number_input(key="npk_N").value, at.number_input(key="npk_P").value,
            at.number_input(key="npk_K").value) == (200.0, 8.0, 280.0)
    # The user can correct a value before predicting
    at.number_input(key="npk_P").set_value(18.0)
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    assert "Suggested crop: **Rice**" in at.success[-1].value
