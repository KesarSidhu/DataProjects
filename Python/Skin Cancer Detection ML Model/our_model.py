import streamlit as st

def our_model():
    st.title("Machine Learning Model")
    # Model overview

    st.write("\n")
    st.image("Images/ML.jpeg", width=300)
    st.write("\n")

    st.subheader("Model Overview")
    st.write("""
    This machine learning model is designed to predict skin cancer from images of skin lesions. This tool was created with the intention of helping people detect cancer earlier and more conveniently since the difference of a few days could save a life.
    
    Our model is built off of the pretrained ResNet model to achieve the predictions. To train the model, we have provided it with a data set that takes in pictures of various skin lesions and teaches it to correctly identify which ones are cancerous.
    """)

    # Data description
    st.subheader("Data Description")
    st.write("""
    The data used for training the model comes from National Library of Medicine. The model was trained on more than 1800 images from the dataset and can diagnose the following 6 types of skin cancers:
    - Actinic Keratosis (ACK)
    - Basal cell carsinoma (BCC)
    - Malignant melanoma (MEL)
    - Melanocytic Nevus of Skin (NEV)
    - Squamoud Cell carcinoma (SCC)
    - Seborrheic Keratosis (SEK)
    """)

    # Data preprocessing
    st.subheader("Data Preprocessing")
    st.write("""
    Before training the model, the data underwent several preprocessing steps:
    1. Removed corrupt fails and files that failed to open.
    2. Removed blank images
    3. Resized all images to be the same size
    4. Categorical variables were encoded.
    """)

    # Model usage
    st.subheader("Model Usage")
    st.write("""
    To use this model for predictions, follow these steps:
    1. Upload an image of a skin lesion on the home page of this website.
    2. Submit the image.
    3. View the results.
    4. Consider consulting the doctor based on the results.

    Here is an example of how to use the model in Python:
    """)

    # Sample code snippet
    st.code("""
    import pandas as pd
    import cv2

    # Load the model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))

    # Prepare the input data
    img_path = '/content/drive/Shareddrives/GDSC Link/Data/imgs_part_3/PAT_1516_1765_530.png'

    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))

    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    # Make a prediction
    result = loaded_model_imageNet.predict(x)
    p = list((result*100).astype('int'))
    pp=list(p[0])
    index = pp.index(max(pp))
    name_class = ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']

    # Display the result
    print(name_class[index])    
    
    """, language='python')

    st.write("""
    For more details on the model and its usage, please refer to the [documentation](https://github.com/srilakshmipanda/GDSC-Link).
    """)

    # Footer
    st.write("___")
    st.write("© 2024 GDSC Link. All rights reserved.")