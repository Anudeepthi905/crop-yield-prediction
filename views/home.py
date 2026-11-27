"""Home page."""
import streamlit as st

from core import IMAGES_DIR
from ui import ABOUT_NOTE

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
