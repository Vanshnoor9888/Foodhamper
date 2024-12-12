import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset with a specified encoding
# data = pd.read_csv('mergedfoodandclients.csv', encoding='latin1')

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')

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
# Function to predict using SARIMA and plot with actual data
def predict_and_compare(start_date, days, actual_data=None):
    """
    Predict the total food hampers needed for a specified number of days and compare with actual data.
    """
    try:
        # Generate exogenous variables
        future_exog = generate_exog(start_date, days)

        # Forecast using SARIMA model
        predictions = sarima_model.forecast(steps=days, exog=future_exog)

        # Create a DataFrame for predictions
        forecast_dates = future_exog.index
        prediction_df = pd.DataFrame({"Date": forecast_dates, "Predicted Hampers": predictions})

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_dates, predictions, label="Predicted", marker="o", linestyle="--")

        if actual_data is not None:
            # Ensure actual data aligns with forecast dates
            actual_data = actual_data.loc[forecast_dates]
            ax.plot(forecast_dates, actual_data, label="Actual", marker="o", color="red")

        ax.set_xlabel("Date")
        ax.set_ylabel("Food Hampers")
        ax.set_title("SARIMA Model Forecast vs Actual")
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return prediction_df, fig
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
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
# Streamlit application
# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Food Hamper Forecasting")

    # Subsection: SARIMA Model for Food Hampers
    st.subheader("Food Hamper Forecasting (SARIMA Model)")

    # Input for start date
    start_date = st.date_input("Select the start date:", datetime.today())

    # Input for the number of days to forecast
    days = st.number_input("Enter the number of days to forecast:", min_value=1, step=1, value=1)

    # Option to upload actual data
    actual_file = st.file_uploader("Upload actual data (CSV with 'Date' and 'Actual Hampers'):", type=["csv"])

    if st.button("Predict and Compare"):
        # Load actual data if uploaded
        actual_data = None
        if actual_file:
            actual_df = pd.read_csv(actual_file, parse_dates=["Date"])
            actual_data = actual_df.set_index("Date")["Actual Hampers"]

        # Call the prediction function with the selected start date and number of days
        predictions_df, fig = predict_and_compare(start_date.strftime("%Y-%m-%d"), int(days), actual_data)

        if predictions_df is not None:
            st.pyplot(fig)
            st.write("### Forecasted Food Hampers")
            st.write(predictions_df)
            total_hampers = predictions_df["Predicted Hampers"].sum()
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
