# Crop Suggestor and Yield Predictor

Upload a photo of an Indian Soil Health Card and get a crop suggestion (Rice, Maize, Groundnut or Redgram) with an estimated yield, based on the soil's Nitrogen, Phosphorus and Potassium (NPK) values.

**Try it:** https://crop-yield-prediction-npk.streamlit.app/

<p>
  <img src="docs/screenshots/predict-result-desktop.png" alt="Predict page on a desktop: N, P and K inputs on the left; on the right, the recommended crop (Maize, 29.87) with a chart ranking all four crops" width="70%">
  <img src="docs/screenshots/predict-result-phone.png" alt="Predict page on a phone, scrolled to the recommendation card after pressing Predict" width="26%">
</p>

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

3. Open http://localhost:8501/predict, choose **Try a sample**, then **Load sample**.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover OCR on every sample card (including enlarged and shrunken images), value parsing and the models, plus the UI with Streamlit's `AppTest`: every page, the empty state, blocked all-zero input, the results card, results surviving reruns, a card being read only once, and friendly recovery when reading or predicting fails. OCR tests are skipped if Tesseract isn't installed.

Browser tests check every page at phone (360 px), tablet (768 px), laptop (1024 px) and desktop (1440 px) widths: no sideways scrolling, no word broken across lines in headings, 16 px inputs, 44 px tap targets on phones, the result scrolled into view on a phone, and two columns from 1024 px. They start the app themselves:

```bash
playwright install chromium
pytest -m e2e                       # browser tests
SAVE_SCREENSHOTS=1 pytest -m e2e    # also refreshes docs/screenshots/
```

## UX decisions

- **Real URLs.** Pages use `st.navigation`, so Home, Predict (`/predict`) and About (`/about`) can be linked to and the back button works.
- **One form, no flicker.** N, P and K sit in one `st.form`: typing doesn't rerun the app, and Enter submits. Labels carry the unit (kg/ha) and help text gives the training range.
- **Validate, don't guess.** All-zero input is blocked with an explanation. Values outside the training data still predict but are flagged as less reliable, because real cards often fall outside it.
- **OCR is a suggestion, not a decision.** A card fills the same form, so people review the values before predicting. Each image is read once. If reading fails, the message says what went wrong, how to fix it, and offers a one-click switch to typing.
- **A result with hierarchy.** The recommendation, its yield and its lead over the runner-up come first; a chart ranks all four crops; exact values sit in an expander. Numbers show 2 decimals.
- **Results don't vanish.** The last result is kept across reruns, with a note if the inputs change after predicting.
- **Every state is designed.** Empty, loading, partial-read and error states each have a message; visitors never see a traceback.
- **Colour has one job.** Brand green marks only the primary action and the winning crop, which is also labelled "(recommended)", so colour is never the only cue.
- **Phone first, then desktop.** On a phone, the result is scrolled into view and focused after Predict (otherwise it lands below the fold and people press again), the three result numbers stay on one row so the chart stays on screen, inputs use 16 px text so iPhones don't zoom in, tap targets are at least 44 px, and headings scale without breaking words. From 1024 px, inputs and result sit side by side, with the result kept in view while scrolling. The few CSS rules Streamlit needs for this live in one documented block in `ui.py` and are guarded by the browser tests.
- **A familiar phone menu.** On phones the navigation sits behind a hamburger (☰) button, a 44 px tap target with a screen-reader label. The drawer has a clear ✕, dims the page behind it, and closes itself when a page is chosen, on a tap outside it, or on Escape. From 768 px the links sit in a top bar instead.
- **Old results fade.** If the inputs change after predicting, the old result is faded under a note, so it can't be mistaken for the current one.
- **Card reading checks itself.** Each value is read several ways (enlarged image, grid lines removed, and a digits-only re-read of the number) and the readings vote. Across 3 Tesseract models and 3 image sizes this took the reader from 120 to 132 of 135 values right, with no wrong values left, only blanks for the user to type.
- **Honest units.** The dataset doesn't state the yield unit, so yields are plain numbers; the unit is one constant (`ui.YIELD_UNIT`) if it's ever confirmed.

## Deploy on Streamlit Community Cloud

1. On https://share.streamlit.io, create an app from this repo with branch `main` and main file path `app.py`.
2. Under **Advanced settings**, choose **Python 3.11**.

`packages.txt` installs Tesseract and `requirements.txt` installs the Python packages. No secrets are needed.

## Limitations

- **OCR works best on clear, straight-on images of the "Soil Test Results" table.** Blurry, angled or differently laid-out cards may not be read; the app then asks for the values to be typed in.
- **Small dataset:** each crop has 225 rows of NPK and yield data, and its model trains on 157–180 of them (the rest are held out for testing). Only N, P and K are used (no rainfall, temperature, soil type or region). Predictions are estimates, not agronomic advice.
- **The yield unit isn't known.** Neither the dataset nor the paper states it, so the app shows yields as plain numbers.
- **The suggestion compares raw predicted yields across crops.** Crops with naturally higher yield figures (Maize in this data) are suggested for most inputs.
- English-language cards only.

## Project structure

```
app.py                  Entry point: page navigation and footer
views/                  The pages: home.py, predict.py, about.py
ui.py                   Shared UI: form, validation, card reading flow, results card, copy
core.py                 Models and OCR (no UI code)
data/                   Training data per crop (N, P, K, Yield)
Images/                 Crop photos on the home page
samples/                Sample Soil Health Cards for "Try a sample"
tools/make_sample_cards.py  Generates the three synthetic sample cards
tests/                  pytest suite (tests/e2e: browser tests at four screen widths)
docs/screenshots/       Screenshots taken by the browser tests
.streamlit/config.toml  Theme and upload limit
packages.txt            System packages for Streamlit Cloud (Tesseract)
requirements.txt        Python dependencies (pinned)
```