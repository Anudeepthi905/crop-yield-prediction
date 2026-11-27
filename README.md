# Crop Suggestor and Yield Predictor

Upload a photo of an Indian Soil Health Card and get a crop recommendation (Rice, Maize, Groundnut or Redgram) with an estimated yield, based on the soil's Nitrogen, Phosphorus and Potassium (NPK) values.

**Live app:** https://crop-yield-prediction-npk.streamlit.app/
**Paper:** [Crop Yield Prediction Using Machine Learning Algorithms](https://doi.org/10.36287/setsci.17.1.0040)

**Tech:** Python · Streamlit · scikit-learn · OpenCV · Tesseract OCR · pytest · Playwright · GitHub Actions

Started in 2023 as a group academic project at GNITS, which led to the paper above. I later rebuilt it: cleaner code, a more reliable card reader, a responsive UI, tests and a live deployment.

## How it works

1. **Read the card:** OpenCV prepares the image and Tesseract reads it; the N, P and K values are filled into a form. No card? Load one of five sample cards.
2. **Check the values:** the user reviews or corrects them, or types them in.
3. **Predict:** one regression model per crop (SVR for Rice, Random Forest for Redgram, Gradient Boosting for Maize and Groundnut) estimates its yield, and the highest is recommended.

## Highlights

- **A card reader that checks itself.** Each value is read several ways and the readings vote; a number is never taken from a range like "120-280". Across 3 Tesseract models and 3 image sizes: 132 of 135 values right, none wrong.
- **Tested on the same Linux as production.** A GitHub Actions check runs the reader on Debian 12 and 13, the way Streamlit Cloud installs Tesseract, and must pass before `main` can change.
- **Designed for phones and desktops.** One form with validation, the result scrolled into view after Predict, a hamburger menu on phones, two columns from 1024 px, and every empty, loading and error state handled.
- **Honest output.** Values outside the training data are flagged, and the yield is shown without a unit because the dataset doesn't state one.

## Run locally

Requires Python 3.11 and [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (`brew install tesseract` / `sudo apt install tesseract-ocr`). Typing values in works without Tesseract.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                     # 34 unit and UI tests
playwright install chromium && pytest -m e2e   # browser tests at 360, 768, 1024 and 1440 px
python tools/ocr_benchmark.py              # card-reader accuracy on every sample card
```

## Limitations

- Trained on a small dataset (225 rows per crop) with only N, P and K; no weather, soil type or region. Predictions are estimates, not agronomic advice.
- The recommendation compares raw yields across crops, so Maize, with the highest yield figures, wins most inputs.
- OCR works best on clear, straight-on photos of the "Soil Test Results" table, in English.

## Project structure

```
app.py      Entry point and page navigation      views/    Home, Predict and About pages
ui.py       UI components and styles             core.py   Models and OCR (no UI code)
data/       Training data per crop               samples/  Sample Soil Health Cards
tests/      Unit, UI and browser tests           tools/    OCR benchmark, sample-card generator
```
