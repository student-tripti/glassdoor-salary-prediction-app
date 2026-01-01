import pickle
import numpy as np
import streamlit as st

with open("ridge_salary_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Glassdoor Salary Prediction App")

min_salary = st.number_input("Minimum Salary", value=50)
company_size = st.selectbox(
    "Company Size (Encoded)",
    [1, 2, 3, 4, 5, 6, 7]
)

if st.button("Predict Salary"):
   input_data = np.array([[
    min_salary,        # Feature 1
    company_size,      # Feature 2
    0,                 # Feature 3 (dummy)
    0,                 # Feature 4 (dummy)
    0,                 # Feature 5 (dummy)
    0                 
]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Average Salary: ${prediction[0]:.2f}K")
