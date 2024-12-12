import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset with a specified encoding
data = pd.read_csv('mergedfoodandclients.csv', encoding='latin1')

# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')

# Function to generate exogenous variables
def generate_exog(days):
    """
    Generate exogenous values for the specified number of days.
    Replace this with your logic to fetch or estimate exog variables.
    """
    future_exog = {
        "scheduled_pickup": [100 + i * 2 for i in range(days)],
        "scheduled_pickup_lag_7": [90 + i for i in range(days)],
        "scheduled_pickup_lag_14": [80 + i for i in range(days)],
    }
    return pd.DataFrame(future_exog)

# Function to predict using SARIMA and plot
def predict_for_days(days):
    """
    Predict the total food hampers needed for a specified number of days and plot.
    """
    try:
        # Generate exogenous variables
        future_exog = generate_exog(days)

        # Forecast using SARIMA model
        predictions = sarima_model.forecast(steps=days, exog=future_exog)

        # Plot the predictions
        time_steps = np.arange(1, days + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_steps, predictions, label="Forecast", marker="o")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Predicted Value")
        ax.set_title("SARIMA Model Forecast")
        ax.legend()
        ax.grid(True)

        return predictions, fig
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

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
    hello = ''' Here’s a concise breakdown of steps we have done:
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
    days = st.number_input("Enter the number of days to forecast:", min_value=1, step=1, value=1)

    if st.button("Predict Food Hampers"):
        predictions, fig = predict_for_days(int(days))
        if fig:
            st.pyplot(fig)
            total_hampers = sum(predictions)
            st.success(f"For {days} days, you will need approximately {int(total_hampers)} food hampers.")
# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Data visualizations", "ML Modeling"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()

if __name__ == "__main__":
    main()
