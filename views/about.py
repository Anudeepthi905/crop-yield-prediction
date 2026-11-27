"""About page: why the app exists, how it works, its limits, and where it came from."""
import streamlit as st

from ui import PAPER_URL, REPO_URL

st.title("About")
st.write("A soil test tells you what's in your soil, but not which crop will do best in it. This app turns the "
         "N, P and K values on a Soil Health Card into a crop recommendation.")

st.subheader("How it works")
st.markdown(
    "- **Reading the card:** OpenCV prepares the image and Tesseract reads the text. The app picks out the N, P "
    "and K values and fills them in for you to check.\n"
    "- **Predicting:** one model per crop estimates its yield: SVR for Rice, Random Forest for Redgram, Gradient "
    "Boosting for Maize, and Histogram-based Gradient Boosting for Groundnut.\n"
    "- **Recommending:** the crop with the highest estimated yield is recommended.")

st.subheader("Limitations")
st.markdown(
    "- The models were trained on a small dataset (225 samples per crop) using only N, P and K: no weather, "
    "soil type or region.\n"
    "- The dataset doesn't state the yield unit, so yields are shown as plain numbers.\n"
    "- Crops with naturally higher yield figures, especially Maize, are recommended for most inputs.\n"
    "- Predictions are estimates, not agronomic advice.")

st.subheader("Background")
st.write(
    "This project began in 2023 as a group academic project at GNITS with my classmates, which led to the "
    f"published paper [*Crop Yield Prediction Using Machine Learning Algorithms*]({PAPER_URL}). I later "
    "revisited it to polish the code, improve the image reading, add tests and put it online as a live app.")
st.markdown(f"[Source code on GitHub]({REPO_URL})")
