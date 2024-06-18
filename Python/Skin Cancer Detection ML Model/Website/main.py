import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import os, glob
from PIL import Image

import pickle
import cv2
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from our_model import our_model 
from meet_the_team import meet_the_team
from history import history

from flask import Flask, render_template

# Set the color scheme
PRIMARY_COLOR = "#81D4FA"
SECONDARY_COLOR = "#A5D6A7"
ACCENT_COLOR = "#FFE0B2"
TEXT_COLOR = "#616161"
BACKGROUND_COLOR = "#FAFAFA"

# GLOBAL variables

def make_prediction():    
    name_class = ['BCC  --  Basal cell carcinoma',
                  'ACK  --  Actinic Keratosis', 
                  'NEV  --  Melanocytic Nevus of Skin', 
                  'SEK  --  Seborrheic Keratosis', 
                  'SCC  --  Squamous Cell carcinoma', 
                  'MEL  --  Malignant melanoma']
    uploaded_file = st.session_state.file
    loaded_model = st.session_state.model

    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        result = loaded_model.predict(x)
        p = list((result*100).astype('int'))
        pp=list(p[0])
        index = pp.index(max(pp))
        st.subheader(name_class[index])
        st.session_state.file = None

def upload():
    st.title('SpotCheckAI', anchor=False)
    st.markdown(
        f"""
        <style>
        .title h1 {{
            color: {PRIMARY_COLOR};
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write(
        "Welcome to the skin lesion detection app! Upload an image of a skin lesion and we'll analyze it for you.",
        unsafe_allow_html=True
    )

    # Upload image
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', width=300)
        st.session_state.file = uploaded_file

    if st.button("Evaluate image"):
        if uploaded_file is None:
            st.subheader("Error: Please upload an image")
        else:
            st.session_state.page = "buffer"
            st.experimental_rerun()

    return uploaded_file
    
def buffer():
        st.title("Your results: ")
        if st.button("Show results"):
            st.session_state.page = "results"
            st.experimental_rerun()

        st.subheader("Disclaimer: ")
        st.write("This model is not fully accurate and we only intend this to be a preliminary prediction. Please consult a doctor if there are any concerns.")

def results():
    st.title("Your results: ")

    # subheader 1
    st.write("Based on our model,  you may have ")
    make_prediction()

    # subheader 2
    st.write("We understand this may be concerning news. We advise that you go to a medical professional to get checked out as soon as possible. It is important to follow their recommendations, which may include further diagnostic tests,  consultations with specialists, and  discussing potential treatment options. Early detection and intervention are crucial for the best possible outcomes.")

    if st.button("Back to Home"):
        st.session_state.page = "upload"
        st.experimental_rerun()



# app = Flask(__name__)

# @app.route('/results')
# def results():
#     # subheader 1
#     section1_data = [
#         {"description": "Based on our model,  you may have cancer."}
#     ]
#     # subheader 2
#     section2_data = [
#         {"description": "We understand this may be concerning news. We advise that you go to a medical professional to get checked out as soon as possible. It is important to follow their recommendations, which may include further diagnostic tests,  consultations with specialists, and  discussing potential treatment options. Early detection and intervention are crucial for the best possible outcomes."}
#     ]
#     return render_template('results.html', section1_data=section1_data, section2_data=section2_data)

def home():
    # loaded model
    if 'model' not in st.session_state:
        loaded_model = load_model('/Users/kesarsidhu/VS Code/GDSC/Streamlit (Cancer Detection)/Main Files/Scripts/trained_model.h5')
        st.session_state.model = loaded_model

    # input image
    if 'file' not in st.session_state:
        st.session_state.file = None

    # Initialize session state if not already done
    if 'page' not in st.session_state:
        st.session_state.page = "upload"

    # Page routing logic
    if st.session_state.page == "upload":
        upload()
    elif st.session_state.page == "buffer":
        buffer()
    elif st.session_state.page == "results":
        results()

    # Footer
    st.write("___")
    st.write("© 2024 GDSC Link. All rights reserved.")

def main():
    st.markdown(
        f"""
        <style>
        .css-18e3th9 {{
            background-color: {BACKGROUND_COLOR};
        }}
        .css-1d391kg {{
            color: {TEXT_COLOR};
            font-family: 'Arial', sans-serif;
        }}
        .css-1v3fvcr {{
            color: {TEXT_COLOR};
            font-family: 'Arial', sans-serif;
        }}
        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 5px;
            font-family: 'Arial', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set page title and description
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Our Model", "About Us", "History"],
            icons=["house", "gear", "people-fill", "clock-history"],
            menu_icon="cast",
            default_index=0,
            styles={
                "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
                "nav-link": {"font-family": "Arial, sans-serif", "color": TEXT_COLOR}
            }
        )


    # Defining each page
    if selected == "Home":
        home()

    elif selected == "Our Model":
        our_model()

    elif selected == "About Us":
        meet_the_team()

    elif selected == "History":
        history()

if __name__ == '__main__':
    main()


#     [theme]
# primaryColor="#76bdb1"
# backgroundColor="#e3e3dc"
# secondaryBackgroundColor="#5b5d6f"
# textColor="#0077e5"
