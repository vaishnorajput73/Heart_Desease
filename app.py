import streamlit as st
import pandas as pd
import joblib

import os
import joblib

import os
import joblib

MODEL_DIR = r"C:\Users\Vaishno Rajput\Downloads"

model = joblib.load(os.path.join(MODEL_DIR, "model_exang.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_exang.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns_exang.pkl"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_exang.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler_exang.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns_exang.pkl"))

print(os.path.exists(os.path.join(BASE_DIR, "model_exang.pkl")))
print(os.path.exists(os.path.join(BASE_DIR, "scaler_exang.pkl")))
print(os.path.exists(os.path.join(BASE_DIR, "columns_exang.pkl")))

import os
print(os.getcwd())
print(os.listdir())
model = joblib.load(r"C:\Users\Vaishno Rajput\Downloads\model_exang.pkl")

'''BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_exang.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler_exang.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns_exang.pkl"))'''

st.title("Heart Stroke Prediction by Akarsh")
st.markdown("Provide the following details")

age = st.slider("Age", 18,100,40)
sex = st.selectbox("Sex", ['Male', 'Female'])
exang = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
restecg = st.number_input("Resting Electrocardiographic Results (0, 1, 2)", min_value=0, max_value=2, step=1)

if st.button("Predict"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'exang': [exang],
        'restecg': [restecg]
    })

    # One-hot encode the categorical variables
    input_data = pd.get_dummies(input_data, columns=['sex', 'exang'])

    # Ensure all columns are present
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[columns]

    # Scale the features
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.error("The model predicts that the patient is at risk of heart stroke.")
    else:
        st.success("The model predicts that the patient is not at risk of heart stroke.")
