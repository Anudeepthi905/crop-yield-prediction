"""Crop Suggestor and Yield Predictor (Streamlit app).

Run with:  streamlit run app.py
"""
import pandas as pd
import pytesseract
import streamlit as st

from core import (ALLOWED_TYPES, IMAGES_DIR, MAX_UPLOAD_MB, SAMPLES, SAMPLES_DIR,
                  predict_yields, read_npk_from_bytes)

YIELD_UNIT_NOTE = "Yields are in the units of the source dataset, which doesn't specify them."

PAPER_URL = "https://doi.org/10.36287/setsci.17.1.0040"
REPO_URL = "https://github.com/Anudeepthi905/crop-yield-prediction"

ABOUT_NOTE = (
    "This app reads the **Nitrogen (N), Phosphorus (P) and Potassium (K)** values from a Soil Health Card "
    "(or takes them typed in), predicts the yield of Rice, Maize, Groundnut and Redgram, and suggests the crop "
    "with the highest predicted yield.\n\n"
    "It is an **academic project**: the models were trained on a small dataset, so predictions are "
    "**estimates only** and not agronomic advice."
)


def home():
    st.markdown("<h1 style='color: green;'>Welcome to the Crop Suggestor and Yield Predictor</h1>",
                unsafe_allow_html=True)
    st.write(
        "The aim is to build crop yield prediction model for a given historical data on NPK values and other environmental factors that affect crop yield, the goal is to develop a model that can accurately predict crop yield based on NPK values. The model should able to take into account the interactions between NPK values and provide accurate predictions for different regions and crop types.")

    st.write(
        "The site will take npk values as input. The user can enter these values manually or extract these values from the image uploaded. The model calculates yields based on npk values and whichever crop produces the highest yield is suggested to the user with the estimated yield. The target user of the site our predominately Farmers, Agricultural Research Centers, etc.")

    st.info(ABOUT_NOTE)

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


def algo(n, p, k):
    py = predict_yields(n, p, k)
    best_crop, best_yield = max(py.items(), key=lambda item: item[1][0])
    st.subheader("Crop Suggestion and Yield Prediction")
    st.success(f"Suggested crop: **{best_crop}**, estimated yield **{best_yield[0]:.2f}**")
    with st.expander("Estimated yield for every crop"):
        table = pd.DataFrame(
            {"Crop": list(py), "Estimated yield": [round(float(v[0]), 2) for v in py.values()]})
        st.dataframe(table.sort_values(table.columns[1], ascending=False), hide_index=True)
    st.caption(f"{YIELD_UNIT_NOTE} Estimates from models trained on a small academic dataset. "
               "Not agronomic advice.")


# ---------------------------------------------------------------------------
# Predictor page
# ---------------------------------------------------------------------------
INPUT_UPLOAD = "Upload a Soil Health Card"
INPUT_SAMPLE = "Try a sample card"
INPUT_MANUAL = "Enter values manually"


def fill_values_from_image(image_bytes, source_id):
    """OCR the image, show it, and pre-fill the N/P/K inputs (once per new image)."""
    try:
        with st.spinner("Reading values from the card..."):
            found = read_npk_from_bytes(image_bytes)
    except pytesseract.TesseractNotFoundError:
        st.error("The OCR engine (Tesseract) isn't available here, so values can't be read from images. "
                 "Please type them below.")
        return
    if found is None:
        st.error("This file couldn't be opened as an image. Please upload a JPG or PNG photo or screenshot "
                 "of the card, or type the values below.")
        return
    st.image(image_bytes, width=520)

    if st.session_state.get("filled_from") != source_id:
        for key, value in found.items():
            st.session_state[f"npk_{key}"] = value
        st.session_state["filled_from"] = source_id

    names = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)"}
    missing = [names[key] for key in names if key not in found]
    if not missing:
        st.success("Values read from the card. Please check them below and correct any mistakes.")
    elif len(missing) < 3:
        st.warning(f"Couldn't read {', '.join(missing)} from this image. Please type the missing values below.")
    else:
        st.warning("Couldn't read the values from this image. Try a clearer, straight-on photo of the "
                   "\"Soil Test Results\" table, or type the values below.")


def upload():
    st.title("Crop Yield Predictor")
    st.info(ABOUT_NOTE)

    for key in ("npk_N", "npk_P", "npk_K"):
        st.session_state.setdefault(key, 0.0)

    st.subheader("1. Provide your soil's N, P and K values")
    option = st.radio("Input method", (INPUT_UPLOAD, INPUT_SAMPLE, INPUT_MANUAL), horizontal=True)

    if option == INPUT_UPLOAD:
        uploaded_file = st.file_uploader(
            "Photo or screenshot of the card's \"Soil Test Results\" table",
            type=ALLOWED_TYPES, help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB.")
        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"That file is {uploaded_file.size / 2**20:.1f} MB. Please upload an image under "
                         f"{MAX_UPLOAD_MB} MB.")
            else:
                image_bytes = uploaded_file.getvalue()
                fill_values_from_image(image_bytes, f"upload:{uploaded_file.name}:{len(image_bytes)}")

    elif option == INPUT_SAMPLE:
        st.caption("No card handy? Pick a sample: a real card (full or cropped) or one of three synthetic ones.")
        choice = st.selectbox("Sample card", list(SAMPLES))
        if st.button("Try a sample"):
            st.session_state["active_sample"] = choice
            st.session_state["filled_from"] = None  # re-fill the values even if this sample was used before
        active = st.session_state.get("active_sample")
        if active == choice:
            fill_values_from_image((SAMPLES_DIR / SAMPLES[active]).read_bytes(), f"sample:{active}")

    st.subheader("2. Check the values (kg/ha)")
    col_n, col_p, col_k = st.columns(3)
    n = col_n.number_input("Nitrogen (N)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_N")
    p = col_p.number_input("Phosphorus (P)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_P")
    k = col_k.number_input("Potassium (K)", min_value=0.0, max_value=10000.0, step=0.01, key="npk_K")

    st.subheader("3. Predict")
    if st.button("Predict", type="primary"):
        if n == 0 and p == 0 and k == 0:
            st.warning("Please enter the N, P and K values first.")
        else:
            algo(n, p, k)


def about():
    st.title("About")

    st.header("Motivation")
    st.write(
        " Food production and prediction is getting depleted due to unnatural climatic changes, which will adversely affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar in forecasting the future crops. This way it helps the farmer in such a way to guide them for sowing the reasonable crops by deploying machine learning, one of the advanced technologies in crop prediction.")

    st.header("Objectives")
    st.write(
        "Our objective is to develop a model that would accurately suggest crop and predict yield in order to help farmers take smart decisions in crop selection. We integrated the model to an easy to use interface .")
    st.header("Background")
    st.write(
        "This project began in 2023 as a group academic project at GNITS with my classmates, which led to the "
        f"published paper [*Crop Yield Prediction Using Machine Learning Algorithms*]({PAPER_URL}). I later "
        "revisited it to polish the code, improve the image reading, add tests and put it online as a live app.")
    st.write(f"Source code: {REPO_URL}")


def main():
    st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")
    menu = ["Home", "Crop Yield Predictor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    st.sidebar.caption("Academic project. Predictions are estimates, not agronomic advice.")

    if choice == "Home":
        home()
    elif choice == "Crop Yield Predictor":
        upload()
    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
