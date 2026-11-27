"""Crop Suggestor and Yield Predictor (Streamlit app).

Run with:  streamlit run app.py
Each page lives in views/ and gets its own URL (/, /predict, /about), so links and the back button work.
"""
import streamlit as st

from ui import REPO_URL, apply_styles

st.set_page_config(page_title="Crop Yield Predictor", page_icon=":material/eco:", layout="centered")
apply_styles()

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/predict.py", title="Predict", icon=":material/eco:", url_path="predict"),
    st.Page("views/about.py", title="About", icon=":material/info:", url_path="about"),
]
st.navigation(pages, position="top").run()

st.divider()
st.caption(f"Academic project. Predictions are estimates, not agronomic advice. [Source code]({REPO_URL})")
