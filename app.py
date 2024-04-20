import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the function to make predictions
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# Define the Streamlit app
def main():
    st.title("Crop Recommendation System")
    st.sidebar.title("Crop Recommendation System")

    st.sidebar.markdown("This app predicts the recommended crop based on input parameters.")

    N = st.sidebar.slider("Nitrogen (N)", 0, 150, 75)
    P = st.sidebar.slider("Phosphorous (P)", 0, 150, 50)
    K = st.sidebar.slider("Potassium (K)", 0, 150, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    ph = st.sidebar.slider("pH Value", 3.0, 10.0, 6.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 400.0, 200.0)

    result = ""

    if st.sidebar.button("Predict"):
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"The recommended crop is: {result}")

if __name__ == "__main__":
    main()
