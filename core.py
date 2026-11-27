"""Model and OCR logic for the Crop Yield Predictor (no UI code).

Moved verbatim from app.py; the Streamlit pages import from here.
"""
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Resolve every file relative to this script, so the app works from any working directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "Images"
SAMPLES_DIR = BASE_DIR / "samples"

DEFAULT_TESSERACT_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Upload limits (Streamlit's own hard limit is set in .streamlit/config.toml: server.maxUploadSize)
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_UPLOAD_MB = 5

# Sample cards visitors can try without uploading: label -> file in samples/
SAMPLES = {
    "Real card (full screenshot)": "sample_card_real_full.jpg",
    "Real card (cropped to the soil test table)": "sample_card_real_cropped.jpg",
    "Synthetic full card (results table + farmer details)": "sample_card_synthetic_full.png",
    "Synthetic card A": "sample_card_synthetic_maize.png",
    "Synthetic card B": "sample_card_synthetic_rice.png",
}


def configure_tesseract():
    """Point pytesseract at the Tesseract binary.

    Order: TESSERACT_CMD env var, then `tesseract` on PATH, then the default Windows install location.
    """
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd and Path(DEFAULT_TESSERACT_WINDOWS).exists():
        cmd = DEFAULT_TESSERACT_WINDOWS
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


FEATURES = ['N', 'P', 'K']

# One model per crop: (training CSV in data/, model factory, test_size, split random_state)
CROP_MODELS = {
    'Redgram': ('Redgram.csv', lambda: RandomForestRegressor(n_estimators=100, random_state=2), 0.2, 2),
    'Rice': ('Rice.csv', lambda: SVR(kernel='linear', C=1.0, epsilon=0.2), 0.2, 2),
    'Maize': ('Maize.csv', lambda: GradientBoostingRegressor(random_state=2), 0.3, 3),
    'Groundnut': ('Groundnut.csv', lambda: HistGradientBoostingRegressor(random_state=2), 0.3, 2),
}


@st.cache_data
def load_training_data(csv_name):
    """Read one crop's training CSV from data/ (cached)."""
    return pd.read_csv(DATA_DIR / csv_name)


@st.cache_resource
def load_models():
    """Train each crop's model once per server process (Streamlit caches the result)."""
    models = {}
    for crop, (csv_name, make_model, test_size, split_seed) in CROP_MODELS.items():
        data = load_training_data(csv_name)
        X_train, _, y_train, _ = train_test_split(
            data[FEATURES], data['Yield'], test_size=test_size, random_state=split_seed)
        models[crop] = make_model().fit(X_train, y_train)
    return models


def predict_yields(n, p, k):
    """Return {crop: predicted_yield_array} for the given NPK values (kg/ha, as on the soil health card)."""
    new_data = pd.DataFrame([[n, p, k]], columns=FEATURES)
    return {crop: model.predict(new_data) for crop, model in load_models().items()}


# ---------------------------------------------------------------------------
# OCR: read N, P, K from a Soil Health Card image
# ---------------------------------------------------------------------------
NUTRIENT_LABELS = {"N": r"Nitrogen", "P": r"Phosph\w*", "K": r"Potass\w*"}
# Values are printed like "305.00". OCR sometimes turns the dot into "," or a space ("36 00").
NUMBER_PATTERN = re.compile(r"(\d{1,4})(?:\s*[.,]\s*|\s)(\d{1,2})\b|(\d{1,4})")
# Readings outside these ranges (kg/ha) are treated as OCR mistakes, e.g. "3600" for "36.00".
PLAUSIBLE_RANGE = {"N": (1, 1500), "P": (0.1, 500), "K": (1, 2000)}


def parse_npk(text):
    """Find N, P and K values in OCR text. Returns a dict with the keys it found."""
    found = {}
    for key, label in NUTRIENT_LABELS.items():
        for line in text.splitlines():
            label_match = re.search(label + r"[^\d\n]*", line, re.IGNORECASE)
            if not label_match:
                continue
            number = NUMBER_PATTERN.search(line, label_match.end())
            if not number:
                continue
            if number.group(1):
                value = float(f"{number.group(1)}.{number.group(2)}")
            else:
                value = float(number.group(3))
            low, high = PLAUSIBLE_RANGE[key]
            if low <= value <= high:
                found[key] = value
                break
    return found


def _ocr(image):
    return pytesseract.image_to_string(image, lang="eng", config="--psm 6")


def _table_region(img):
    """The original approach: OCR the first large, table-shaped region of the image."""
    threshold = int(0.01 * img.shape[0] * img.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > threshold and 1.2 < w / h < 2.5:
            return img[y:y + h, x:x + w]
    return None


def _without_table_lines(img):
    """Upscale, binarize and erase table grid lines, which otherwise confuse Tesseract."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, gray.shape[1] // 15), 1)))
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, gray.shape[0] // 15))))
    ink[(horizontal | vertical) > 0] = 0
    return 255 - ink


def read_npk(img):
    """OCR a soil health card image (BGR numpy array).

    Tries several views of the image and returns the first complete reading, otherwise the
    values it could find (a dict with any of the keys "N", "P", "K").
    """
    configure_tesseract()
    partial = {}
    views = (lambda: _table_region(img), lambda: _without_table_lines(img), lambda: img)
    for make_view in views:
        view = make_view()
        if view is None:
            continue
        found = parse_npk(_ocr(view))
        if len(found) == 3:
            return found
        for key, value in found.items():
            partial.setdefault(key, value)
    return partial


def extract_npk_from_image(img):
    """Returns (n, p, k) floats if all three values were read, otherwise None."""
    found = read_npk(img)
    return (found["N"], found["P"], found["K"]) if len(found) == 3 else None


@st.cache_data(show_spinner=False, max_entries=50)
def read_npk_from_bytes(image_bytes):
    """Decode image bytes and OCR them. Returns None if the bytes are not a readable image."""
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return read_npk(img)
