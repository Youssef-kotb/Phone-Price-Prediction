import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('/home/joo/Desktop/Phone Price Prediction/Deployment/phone_price_model.pkl')
scaler = joblib.load('/home/joo/Desktop/Phone Price Prediction/Deployment/scaler.pkl') 



st.title("📱 Phone Price Predictor")

# Inputs
ppi = st.number_input("Pixels Per Inch (ppi)", min_value=121)

cpu_core = st.number_input("CPU Cores", min_value=1)
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.2)

internal_mem = st.number_input("Internal Memory (GB)", min_value=4)

RearCam = st.number_input("Rear Camera (MP)", min_value=1.3)
Front_Cam = st.number_input("Front Camera (MP)", min_value=0.9)

battery = st.number_input("Battery Capacity (mAh)", min_value=1500)
weight = st.selectbox("Weight",['light','medium','heavy'] )

thickness = st.number_input("Thickness (mm)", min_value=5.1)

Performance_score = cpu_core * cpu_freq

Camera_score = RearCam * 0.7 + Front_Cam * 0.3

# Manually one-hot encode the input
weight_categories = ["Heavy", "Light", "Medium"]
weight_input_dict = {f"weight_{cat}": 1 if weight == cat else 0 for cat in weight_categories}

# Combine all inputs
input_data = {
    "ppi": ppi,
    "cpu core": cpu_core,
    "cpu freq": cpu_freq,
    "internal mem": internal_mem,
    "RearCam": RearCam,
    "Front_Cam": Front_Cam,
    "battery": battery,
    "thickness": thickness,
    "Performance_score": Performance_score,
    "Camera_score": Camera_score,
    **weight_input_dict
}

input_data = pd.DataFrame([input_data])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Price"):
    price = int(model.predict(input_data_scaled))
    st.success(f"💰 Predicted Price: {price} EGP")
