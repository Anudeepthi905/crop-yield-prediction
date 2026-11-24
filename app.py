"""Crop Suggestor and Yield Predictor (Streamlit app).

Run with:  streamlit run app.py
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

DEFAULT_TESSERACT_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract():
    """Point pytesseract at the Tesseract binary.

    Order: TESSERACT_CMD env var, then `tesseract` on PATH, then the default Windows install location.
    """
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd and Path(DEFAULT_TESSERACT_WINDOWS).exists():
        cmd = DEFAULT_TESSERACT_WINDOWS
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def home():
    st.markdown("<h1 style='color: green;'>Welcome to the Crop Suggestor and Yield Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color: black;'>The aim is to build crop yield prediction model for a given historical data on NPK values and other environmental factors that affect crop yield, the goal is to develop a model that can accurately predict crop yield based on NPK values. The model should able to take into account the interactions between NPK values and provide accurate predictions for different regions and crop types.</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<p style='color: black;'>The site will take npk values as input. The user can enter these values manually or extract these values from the image uploaded. The model calculates yields based on npk values and whichever crop produces the highest yield is suggested to the user with the estimated yield in tons per hectare. The target user of the site our predominately Farmers, Agricultural Research Centers, etc.</p>",
        unsafe_allow_html=True)

    image1 = str(IMAGES_DIR / "rice.jpg")
    image2 = str(IMAGES_DIR / "maize.jpg")
    image3 = str(IMAGES_DIR / "groundnut.jpg")
    image4 = str(IMAGES_DIR / "redgram.jpg")

    # create the first container
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, width="stretch")
            st.markdown("<h2 style='color: blue;'>Rice</h2>", unsafe_allow_html=True)
            st.write(
                "Rice is an important crop in Telangana, India, and is widely cultivated across the state. The state government has taken several initiatives to promote rice cultivation and increase the productivity of rice crops. In Telangana, rice is mainly cultivated during the kharif season, which typically starts in June and lasts until October. The state government has introduced several initiatives to support rice farmers during the kharif season, such as the provision of seeds, fertilizers, and other inputs at subsidized rates, as well as the implementation of irrigation projects and other infrastructure developments. Rice crops typically require a large amount of nitrogen, as it is essential for plant growth and needs low phosphorus levels & low potassium levels.")
        with col2:
            st.image(image2, width="stretch")
            st.markdown("<h2 style='color: blue;'>Maize</h2>", unsafe_allow_html=True)
            st.write(
                "Maize is important as a staple food, economic crop, and a source of genetic diversity. Maize plays a crucial role in global agriculture and food security. Maize (also known as corn) is an important cereal crop grown in many parts of the world, including India. It is a warm-season crop that is mainly grown during the kharif season (June to October) in India. The crop is also used as animal feed and is an important source of energy and protein for livestock. Maize crops require a relatively high amount of nitrogen for optimal growth and yield and need low phosphorus levels & low potassium levels. Adequate nutrient availability is crucial for achieving high yields and maintaining the overall health of maize plants.")
    # create the second container
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(image3, width="stretch")
            st.markdown("<h2 style='color: blue;'>Groundnut</h2>", unsafe_allow_html=True)
            st.write(
                "Groundnut (also known as peanut) is an important crop in many parts of the world, including India. In India, groundnut is mainly grown as a kharif crop (during the monsoon season), although it can also be grown as a rabi crop (during the winter season) in some areas. The crop requires well-drained soil with good water-holding capacityIn India, groundnut is an important source of protein and oil, and is used in a variety of food products. The crop also provides a source of income for many small and marginal farmers in rural areas. Groundnut crops require a moderate amount of nitrogen and needs low phosphorus levels & low potassium levels.")
        with col2:
            st.image(image4, width="stretch")
            st.markdown("<h2 style='color: blue;'>Redgram</h2>", unsafe_allow_html=True)
            st.write(
                "Red gram (also known as pigeon pea) is an important pulse crop grown in India, particularly in the states of Maharashtra, Karnataka, Andhra Pradesh, and Telangana. The crop is also known for its nitrogen-fixing ability, which can help improve soil fertility and reduce the need for nitrogen fertilizers. -Redgram crops have a moderate nitrogen requirement, but they can fix atmospheric nitrogen through a symbiotic relationship with soil bacteria and needs low phosphorus levels & low potassium levels.")


FEATURES = ['N', 'P', 'K']

# One model per crop: (training CSV in data/, model factory, test_size, split random_state)
CROP_MODELS = {
    'Redgram': ('Redgram.csv', lambda: RandomForestRegressor(n_estimators=100, random_state=2), 0.2, 2),
    'Rice': ('Rice.csv', lambda: SVR(kernel='linear', C=1.0, epsilon=0.2), 0.2, 2),
    'Maize': ('Maize.csv', lambda: GradientBoostingRegressor(random_state=2), 0.3, 3),
    'Groundnut': ('Groundnut.csv', lambda: HistGradientBoostingRegressor(random_state=2), 0.3, 2),
}


@st.cache_resource
def load_models():
    """Train each crop's model once per server process (Streamlit caches the result)."""
    models = {}
    for crop, (csv_name, make_model, test_size, split_seed) in CROP_MODELS.items():
        data = pd.read_csv(DATA_DIR / csv_name)
        X_train, _, y_train, _ = train_test_split(
            data[FEATURES], data['Yield'], test_size=test_size, random_state=split_seed)
        models[crop] = make_model().fit(X_train, y_train)
    return models


def predict_yields(n, p, k):
    """Return {crop: predicted_yield_array} for the given NPK values (kg/ha, as on the soil health card)."""
    new_data = pd.DataFrame([[n, p, k]], columns=FEATURES)
    return {crop: model.predict(new_data) for crop, model in load_models().items()}


def algo(n, p, k):
    py = predict_yields(n, p, k)
    best_crop, best_yield = max(py.items(), key=lambda item: item[1][0])
    st.write("Crop Suggestion and Yield Prediction")
    st.write(best_crop, best_yield[0], "kg/ha")


def extract_npk_from_image(img):
    """OCR a soil health card image (BGR numpy array). Returns (n, p, k) floats, or None if not found."""
    configure_tesseract()
    # Calculate threshold and aspect ratio based on image size
    img_size = img.shape[0] * img.shape[1]
    threshold = int(0.01 * img_size)
    aspect_ratio_range = (1.2, 2.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract NPK values from contours using OCR
    npk_values = ""
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if area > threshold and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
            roi = img[y:y + h, x:x + w]
            npk_values = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
            if npk_values:
                break

    if not npk_values:
        return None
    # Extract decimal values for N, P, and K using regular expressions
    npk_values = npk_values.strip()
    n_match = re.search(r"Available Nitrogen \(N\) \| (\d+\.\d+)", npk_values)
    p_match = re.search(r"Available Phosphorus \(P\)\| (\d+\.\d+)", npk_values)
    k_match = re.search(r"Available Potassium \(K\) \| (\d+\.\d+)", npk_values)
    if not (n_match and p_match and k_match):
        return None
    return float(n_match.group(1)), float(p_match.group(1)), float(k_match.group(1))


def upload():
    st.title("Crop Yield Predictor")
    st.write("Please select an option to input soil data:")

    option = st.radio("Input Method", ("Upload Soil Health Card Image", "Manual Input"))

    if option == "Upload Soil Health Card Image":
        st.write("Please upload an image of your soil health card")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None:
                st.warning("Could not read the uploaded file as an image. Please enter values manually.")
                st.stop()
            st.image(img, channels="BGR")
            try:
                npk = extract_npk_from_image(img)
            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract OCR is not installed or not found. Install it or set TESSERACT_CMD, "
                         "or use Manual Input.")
                st.stop()
            # If NPK values not found, ask user to input manually
            if npk is None:
                st.warning("Unable to extract values from Image please enter manually")
                st.stop()
            n, p, k = npk
            algo(n, p, k)

    elif option == "Manual Input":
        st.write("Please enter the values of Nitrogen, Phosphorous, and Potassium below.")
        n = st.number_input("Enter Nitrogen value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        p = st.number_input("Enter Phosphorous value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        k = st.number_input("Enter Potassium value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)

        if st.button("Predict"):
            algo(n, p, k)


def about():
    st.title("About us")

    st.header("Motivation")
    st.write(
        " Food production and prediction is getting depleted due to unnatural climatic changes, which will adversely affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar in forecasting the future crops. This way it helps the farmer in such a way to guide them for sowing the reasonable crops by deploying machine learning, one of the advanced technologies in crop prediction.")

    st.header("Objectives")
    st.write(
        "Our objective is to develop a model that would accurately suggest crop and predict yield in order to help farmers take smart decisions in crop selection. We integrated the model to an easy to use interface .")
    st.header("Group Members")
    st.write("3rd Year CSD, Batch-D11")

    members_container = st.container()

    with members_container:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Name: K Anu deepthi")
            st.write("Roll no: 20251A6712")

        with col2:
            st.write("Name: Shreya A")
            st.write("Roll no: 20251A6727")

        with col3:
            st.write("Name: U Sai Amogha")
            st.write("Roll no: 20251A6728")

        with col4:
            st.write("Name: Siri Vennela B")
            st.write("Roll no: 20251A6731")


def main():
    menu = ["Home", "Crop Yield Predictor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        home()
    elif choice == "Crop Yield Predictor":
        upload()
    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
