# Crop Suggestor and Yield Predictor

Upload a photo of an Indian Soil Health Card and get a crop suggestion (Rice, Maize, Groundnut or Redgram) with an estimated yield, based on the soil's Nitrogen, Phosphorus and Potassium (NPK) values.

**Try it:** https://crop-yield-prediction-npk.streamlit.app/

<!-- SCREENSHOT_TODO: add a screenshot or GIF of the app, e.g. ![App screenshot](assets/screenshot.png) -->

**Tech:** Python · Streamlit · scikit-learn · OpenCV · Tesseract OCR · pandas · pytest

**Paper:** *Crop Yield Prediction Using Machine Learning Algorithms*, https://doi.org/10.36287/setsci.17.1.0040

This project began in 2023 as a group academic project at GNITS with my classmates, which led to the published paper above. I later revisited it to polish the code, improve the image reading, add tests and put it online as a live app.

## How it works

```
Soil Health Card image ──► OCR ──► N, P, K values ──► one model per crop ──► crop with the highest predicted yield
                         (OpenCV +       (you can check
                          Tesseract)      and correct them)
```

1. **OCR:** OpenCV cleans up the image (finds the results table, erases grid lines) and Tesseract reads the text. The app then picks out the *Available Nitrogen*, *Phosphorus* and *Potassium* values.
2. **Check:** the values are filled into editable fields, so you can fix any OCR mistakes, or skip the image and type them in.
3. **Predict:** each crop has its own regression model, trained on `data/<Crop>.csv`. The app shows the crop with the highest predicted yield, plus the estimate for every crop.

No Soil Health Card? The app has a **Try a sample** button with five sample cards: a real card (full screenshot, and cropped to the soil test table), and three synthetic ones, including a full-layout card with made-up farmer details.

## Models compared

| Crop      | Model |
|-----------|-------|
| Redgram   | Random Forest Regressor |
| Rice      | Support Vector Regressor (linear kernel) |
| Maize     | Gradient Boosting Regressor |
| Groundnut | Histogram-based Gradient Boosting Regressor |

The models are trained when the app starts, which takes under a second, and cached. This repo doesn't include evaluation metrics; see the paper for the model comparison.

## Run locally

Requires **Python 3.11** and **Tesseract OCR** (only needed for reading images; typing values in works without it).

1. Install Tesseract:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki (the default install path is detected automatically)
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`

   If it's installed somewhere else, set the `TESSERACT_CMD` environment variable to the full path of the executable.

2. Install the Python dependencies and start the app:

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux

   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Open http://localhost:8501, choose **Crop Yield Predictor** in the sidebar, then **Try a sample card**.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover OCR on every sample card, value parsing, the models, every page of the UI, the sample button and a manual prediction. OCR tests are skipped if Tesseract isn't installed.

## Deploy on Streamlit Community Cloud

1. On https://share.streamlit.io, create an app from this repo with branch `main` and main file path `app.py`.
2. Under **Advanced settings**, choose **Python 3.11**.

`packages.txt` installs Tesseract and `requirements.txt` installs the Python packages. No secrets are needed.

## Limitations

- **OCR works best on clear, straight-on images of the "Soil Test Results" table.** Blurry, angled or differently laid-out cards may not be read; the app then asks for the values to be typed in.
- **Small dataset:** each model is trained on about 180 rows of NPK and yield data, and only N, P and K are used (no rainfall, temperature, soil type or region). Predictions are estimates, not agronomic advice.
- **The yield unit isn't known.** Neither the dataset nor the paper states it, so the app shows yields as plain numbers.
- **The suggestion compares raw predicted yields across crops.** Crops with naturally higher yield figures (Maize in this data) are suggested for most inputs.
- English-language cards only.

## Project structure

```
app.py                  Streamlit app (UI, OCR, models)
data/                   Training data per crop (N, P, K, Yield)
Images/                 Crop photos on the home page
samples/                Sample Soil Health Cards for "Try a sample"
tools/make_sample_cards.py  Generates the three synthetic sample cards
tests/                  pytest suite
.streamlit/config.toml  Theme and upload limit
packages.txt            System packages for Streamlit Cloud (Tesseract)
requirements.txt        Python dependencies (pinned)
```