"""Home page: what the app does, one call to action, how it works."""
import streamlit as st
from PIL import Image, ImageOps

from core import IMAGES_DIR


@st.cache_data(show_spinner=False)
def crop_photo(filename):
    """Centre-crop to one shape (3:2) so the four cards line up."""
    with Image.open(IMAGES_DIR / filename) as image:
        return ImageOps.fit(image.convert("RGB"), (600, 400))


CROPS = [
    ("Rice", "rice.jpg", "Telangana's main kharif crop, grown June to October. Needs plenty of nitrogen."),
    ("Maize", "maize.jpg", "A warm-season kharif cereal for food and animal feed. Needs fairly high nitrogen."),
    ("Groundnut", "groundnut.jpg", "An oilseed grown in kharif or rabi on well-drained soil. Needs moderate nitrogen."),
    ("Redgram", "redgram.jpg", "Pigeon pea, a pulse that fixes its own nitrogen. Needs moderate nitrogen."),
]

st.title("Find the best crop for your soil")
st.write("Upload your Soil Health Card, or type in its nitrogen (N), phosphorus (P) and potassium (K) values. "
         "The app estimates the yield of four crops and recommends the highest.")
if st.button("Get a recommendation", type="primary", icon=":material/arrow_forward:"):
    st.switch_page("views/predict.py")

st.subheader("How it works")
step_1, step_2, step_3 = st.columns(3)
with step_1:
    st.markdown("**1. Add your values**")
    st.caption("Upload a card photo, load a sample card, or type N, P and K.")
with step_2:
    st.markdown("**2. Check them**")
    st.caption("Values read from the card are filled in for you to review and correct.")
with step_3:
    st.markdown("**3. Get a recommendation**")
    st.caption("See the best crop, its estimated yield, and how all four crops compare.")

st.subheader("Crops covered")
for column, (name, image, blurb) in zip(st.columns(4), CROPS):
    with column:
        st.image(crop_photo(image), width="stretch")
        st.markdown(f"**{name}**")
        st.caption(blurb)
