import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

# Load the trained LightGBM model
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to predict crop based on input features
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    # Title
    st.title("Crop Recommendation System")

    # Sidebar with input fields
    st.sidebar.header("Input Parameters")

    N = st.sidebar.slider("Nitrogen (N) ratio in soil (kg/ha)", 0.0, 200.0, 100.0)
    P = st.sidebar.slider("Phosphorous (P) ratio in soil (kg/ha)", 0.0, 200.0, 100.0)
    K = st.sidebar.slider("Potassium (K) ratio in soil (kg/ha)", 0.0, 200.0, 100.0)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 50)
    ph = st.sidebar.slider("pH value of the soil", 0.0, 14.0, 7.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1000.0, 500.0)

    # Predict crop based on input features
    prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    crop_list = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
                 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
                 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
                 'rice', 'watermelon']
    predicted_crop = crop_list[prediction]

    # Display prediction
    st.subheader("Predicted Crop:")
    st.write(predicted_crop)

if __name__ == "__main__":
    main()
