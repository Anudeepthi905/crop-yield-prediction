"""Smoke tests. Run from the project root with:  pytest"""
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "sample" / "SHC.jpg"


def tesseract_available():
    app.configure_tesseract()
    return shutil.which(app.pytesseract.pytesseract.tesseract_cmd) is not None


def load_sample():
    # np.fromfile + imdecode handles paths with spaces/non-ASCII on Windows (cv2.imread does not).
    img = cv2.imdecode(np.fromfile(str(SAMPLE_IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, "sample/SHC.jpg could not be decoded"
    return img


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract OCR not installed")
def test_ocr_extracts_npk_from_sample_card():
    assert app.extract_npk_from_image(load_sample()) == (305.0, 36.0, 69.0)


def test_predict_yields_returns_all_crops():
    yields = app.predict_yields(305.0, 36.0, 69.0)
    assert set(yields) == {"Redgram", "Rice", "Maize", "Groundnut"}
    for value in yields.values():
        assert math.isfinite(float(value[0]))


def test_all_referenced_files_exist():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    images = re.findall(r'IMAGES_DIR / "([^"]+)"', source)
    assert images, "expected app.py to reference images"
    for name in images:
        assert (app.IMAGES_DIR / name).is_file(), f"missing image: Images/{name}"
    for csv_name, *_ in app.CROP_MODELS.values():
        assert (app.DATA_DIR / csv_name).is_file(), f"missing data file: data/{csv_name}"


@pytest.mark.parametrize("page", ["Home", "Crop Yield Predictor", "About"])
def test_pages_render_without_errors(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.selectbox[0].select(page).run()
    assert not at.exception


def test_manual_input_prediction():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.selectbox[0].select("Crop Yield Predictor").run()
    at.radio[0].set_value("Manual Input").run()
    at.number_input[0].set_value(305.0)
    at.number_input[1].set_value(36.0)
    at.number_input[2].set_value(69.0)
    at.button[0].click().run()
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "Crop Suggestion and Yield Prediction" in rendered
    assert "kg/ha" in rendered
