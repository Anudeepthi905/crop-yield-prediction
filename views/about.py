"""About page."""
import streamlit as st

from ui import PAPER_URL, REPO_URL

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
