import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/iris_model.pkl")

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Enter flower measurements to predict the species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"Predicted Species: **{prediction[0]}**")
