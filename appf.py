import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from scipy.special import inv_boxcox

# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')

# Load dataset
data = pd.read_csv('mergedfoodandclients.csv', encoding='latin1', parse_dates=['date'])
data.set_index('date', inplace=True)

# Function to generate exogenous variables
def generate_exog(start_date, days):
    """
    Generate exogenous values for the specified number of days.
    Replace this with your logic to fetch or estimate exog variables.
    """
    future_exog = {
        "scheduled_pickup": [100 + i * 2 for i in range(days)],
        "scheduled_pickup_lag_7": [90 + i for i in range(days)],
        "scheduled_pickup_lag_14": [80 + i for i in range(days)],
    }
    return pd.DataFrame(future_exog, index=pd.date_range(start=start_date, periods=days, freq="D"))

# Function to predict using SARIMA and plot
def predict_and_plot(start_date, days, lam):
    try:
        # Generate exogenous variables
        future_exog = generate_exog(start_date, days)

        # Forecast using SARIMA model
        predictions = sarima_model.forecast(steps=days, exog=future_exog)

        # Inverse Box-Cox transformation
        forecast_original = inv_boxcox(predictions, lam)

        # Create DataFrame for forecast results
        forecast_dates = future_exog.index
        prediction_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Hampers (Box-Cox Transformed)": predictions,
            "Predicted Hampers (Original Scale)": forecast_original
        })

        # Plot results
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Plot for Box-Cox transformed data
        ax[0].plot(data.index, data['actual_pickup_boxcox'], label="Actual (Train/Test)", color="blue")
        ax[0].plot(forecast_dates, predictions, label="Forecast (Box-Cox Transformed)", color="orange")
        ax[0].set_title("SARIMA Forecast (Box-Cox Transformed)")
        ax[0].fill_between(
            forecast_dates,
            predictions - 1.96 * np.std(predictions),
            predictions + 1.96 * np.std(predictions),
            color='gray', alpha=0.3, label="Confidence Interval"
        )
        ax[0].legend()
        ax[0].grid(True)

        # Plot for original scale
        ax[1].plot(data.index, data['actual_pickup'], label="Actual (Train/Test)", color="blue")
        ax[1].plot(forecast_dates, forecast_original, label="Forecast (Original Scale)", color="green")
        ax[1].set_title("SARIMA Forecast (Original Scale)")
        ax[1].fill_between(
            forecast_dates,
            forecast_original - 1.96 * np.std(forecast_original),
            forecast_original + 1.96 * np.std(forecast_original),
            color='gray', alpha=0.3, label="Confidence Interval"
        )
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        return prediction_df, fig

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Streamlit application
# Page 1: Dashboard
def dashboard():
    st.subheader("💡 Project Overview:")
    inspiration = '''Project Overview We are collaborating on a machine learning project with a food hamper
    distribution company. The organization has shared their dataset with us and highlighted a number of challenges
    they face, such as resource allocation and meeting rising demand. After analyzing their needs, we identified that predicting 
    the number of food hampers to be distributed in the near future could address several of these challenges. Our project will focus on 
    developing a model to accurately forecast hamper distribution, enabling better planning and resource management for the organization.
    '''
    st.write(inspiration)
    st.subheader("Steps :")
    hello = ''' Here’s a concise breakdown of the steps we have done:
    1. Data Cleaning
    2. Data Visualizations
    3. ML Modelling
    4. Chat Bot
    '''
    st.write(hello)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Data Visualizations")
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/b91808fe-0100-4e7f-94d4-957c4fea0c20/page/AtrGE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Food Hamper Forecasting")

    # Subsection: SARIMA Model for Food Hampers
    st.subheader("Food Hamper Forecasting (SARIMA Model)")

    # Input for start date
    start_date = st.date_input("Select the start date:", datetime.today())

    # Input for the number of days to forecast
    days = st.number_input("Enter the number of days to forecast:", min_value=1, step=1, value=7)

    # Input for Box-Cox lambda value
    lam = st.number_input("Enter the lambda value for Box-Cox transformation:", min_value=0.0, step=0.1, value=1.0)

    if st.button("Predict Food Hampers"):
        # Call the prediction and plotting function
        predictions_df, fig = predict_and_plot(start_date.strftime("%Y-%m-%d"), int(days), lam)

        if predictions_df is not None:
            st.pyplot(fig)
            st.write("### Forecasted Food Hampers")
            st.write(predictions_df)
            total_hampers = predictions_df["Predicted Hampers (Original Scale)"].sum()
            st.success(f"For {days} days starting from {start_date}, "
                       f"you will need approximately {int(total_hampers)} food hampers.")

# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Data visualizations", "ML Modeling"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Data visualizations":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()

if __name__ == "__main__":
    main()
