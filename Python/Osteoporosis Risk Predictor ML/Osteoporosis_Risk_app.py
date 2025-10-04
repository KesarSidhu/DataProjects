import streamlit as st
import pickle
import numpy as np

# Load the decision tree trained model
with open('decision_tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
    
# Title for the app with a little custom styling
st.title("Osteoporosis Risk Prediction")

# Subtitle or introduction text with explanation
st.write("This application predicts the risk of osteoporosis based on various health factors.")

# Section header for the inputs
st.write("Please provide the following information:")

# Gather input for numerical feature with a description
age = st.number_input("Age", min_value = 18, max_value = 90, step = 1, help = "Enter the age of the individual")

# Section for categorical features with organized layout
col1, col2 = st.columns(2)  # Create two columns for a cleaner layout

# Gather categorical input in columns to balance the UI
with col1:
    hormonal_changes = st.selectbox("Hormonal Changes", options = ["Normal", "Postmenopausal"], help = "Select the hormonal status")
    body_weight = st.selectbox("Body Weight", options = ["Normal", "Underweight"], help = "Select the body weight status")
    calcium_intake = st.selectbox("Calcium Intake", options = ["Adequate", "Low"], help = "Select calcium intake level")
    vitamin_d_intake = st.selectbox("Vitamin D Intake", options = ["Insufficient", "Sufficient"], help = "Select Vitamin D intake level")
    
with col2:
    physical_activity = st.selectbox("Physical Activity", options = ["Active", "Sedentary"], help = "Select activity level")
    medications = st.selectbox("Medications", options = ["Corticosteroids", "None"], help = "Select any medications taken")
    prior_fractures = st.selectbox("Prior Fractures", options = ["No", "Yes"], help = "Has the individual had any prior fractures?")
    medical_conditions = st.selectbox("Medical Conditions", options = ["Hyperthyroidism", "None", "Rheumatoid Arthritis"], help = "Select medical conditions")


# Encoding categorical inputs
hormonal_changes = {"Normal": 0, "Postmenopausal": 1}[hormonal_changes]
body_weight = {"Normal": 0, "Underweight": 1}[body_weight]
calcium_intake = {"Adequate": 0, "Low": 1}[calcium_intake]
vitamin_d_intake = {"Insufficient": 0, "Sufficient": 1}[vitamin_d_intake]
physical_activity = {"Active": 0, "Sedentary": 1}[physical_activity]
medications = {"Corticosteroids": 0, "None": 1}[medications]
prior_fractures = {"No": 0, "Yes": 1}[prior_fractures]
medical_conditions = {"Hyperthyroidism": 0, "None": 1, "Rheumatoid Arthritis": 2}[medical_conditions]

# Place all input data into a list for prediction
input_data = [
    int(age),
    int(hormonal_changes),
    int(body_weight),
    int(calcium_intake),
    int(vitamin_d_intake),
    int(physical_activity),
    int(medical_conditions),
    int(medications),
    int(prior_fractures)
]

# Reshape data to 2D array for prediction
input_data = np.array(input_data).reshape(1, -1)

# Prediction section with a button
if st.button("Predict Osteoporosis Risk"):
    
    # Make prediction using the trained model
    prediction = dt_model.predict(input_data)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.write("### Osteoporosis Risk: High Risk")
    else:
        st.write("### Osteoporosis Risk: Low Risk")