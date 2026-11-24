# Crop Suggestor and Yield Predictor

A Streamlit web app that suggests the best crop (Rice, Maize, Groundnut or Redgram) for a field and estimates its yield from the soil's Nitrogen (N), Phosphorus (P) and Potassium (K) values.

**Tech:** Python · scikit-learn · OpenCV · Tesseract OCR · pandas · Streamlit · pytest

## Why

Indian farmers receive a government **Soil Health Card** listing their soil's nutrient levels, but turning those numbers into a crop decision is hard. This app does it in one step: upload a photo of the card and get a crop suggestion with an estimated yield.

## Highlights

- **Reads the card for you:** OpenCV finds the results table and Tesseract OCR extracts the N, P and K values. Manual entry is also available.
- **Compares one model per crop:** four regression models are trained and the crop with the highest predicted yield is suggested.
- **Tested:** pytest covers the OCR on a real sample card, all four models, every page of the UI and a full manual-input prediction.

## How it works

1. **Input:** NPK values typed in, or read from an uploaded Soil Health Card image.
2. **Predict:** each crop's model, trained on `data/<Crop>.csv`, predicts that crop's yield for these NPK values.
3. **Suggest:** the app shows the crop with the highest predicted yield.

The models:

| Crop      | Model                          |
|-----------|--------------------------------|
| Redgram   | Random Forest Regressor        |
| Rice      | Support Vector Regressor (linear) |
| Maize     | Gradient Boosting Regressor    |
| Groundnut | Histogram Gradient Boosting    |

The app suggests the crop with the highest predicted yield.

## Project structure

```
app.py               Streamlit app (UI, models, OCR)
data/                Training data per crop (N, P, K, Yield)
Images/              Crop photos used on the home page
sample/SHC.jpg       Sample Soil Health Card for testing the OCR path
docs/                Project report and presentation
tests/               Smoke tests (pytest)
requirements.txt     Runtime dependencies
```

## Setup

Requires **Python 3.11** (tested), plus **Tesseract OCR** for the image-upload feature.

1. Install Tesseract:
   - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically)
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`

   If Tesseract is installed somewhere else, set the `TESSERACT_CMD` environment variable to the full path of the executable.

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

### Try it with the sample card

1. Choose **Crop Yield Predictor** in the sidebar menu.
2. Select **Upload Soil Health Card Image** and upload `sample/SHC.jpg`.
3. The app reads **N = 305.00, P = 36.00, K = 69.00** from the card and shows the suggested crop and its predicted yield.

You can also pick **Manual Input** and type those same values.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests check OCR on `sample/SHC.jpg` (skipped if Tesseract isn't installed), run the models, render every page and run a manual-input prediction.

## Team

3rd Year CSD, Batch D11: K Anu Deepthi, Shreya A, U Sai Amogha, Siri Vennela B.
