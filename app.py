import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('sarima_model.pkl')

def generate_exog(days):
    future_exog = {
        "scheduled_pickup": [100 + i * 2 for i in range(days)],
        "scheduled_pickup_lag_7": [90 + i for i in range(days)],
        "scheduled_pickup_lag_14": [80 + i for i in range(days)],
    }
    return pd.DataFrame(future_exog)

def predict_for_days(days):
    try:
        future_exog = generate_exog(days)
        predictions = model.forecast(steps=days, exog=future_exog)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

st.title("Food Hamper Prediction App")
st.write("This app predicts the total number of food hampers needed for a specified number of days.")

days = st.number_input("Enter the number of days for prediction:", min_value=1, value=1, step=1)

if st.button("Predict"):
    predictions = predict_for_days(days)
    if predictions is not None:
        time_steps = np.arange(1, days + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(time_steps, predictions, label="Forecast", marker="o")
        plt.xlabel("Time Step")
        plt.ylabel("Predicted Value")
        plt.title("SARIMA Model Forecast")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
