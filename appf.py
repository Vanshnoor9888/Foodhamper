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
    st.subheader("💡 Abstract:")
    inspiration = '''
Data Quality: It is impossible to exaggerate the significance of data quality...
    '''
    st.write(inspiration)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/b91808fe-0100-4e7f-94d4-957c4fea0c20/page/AtrGE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Kijiji Rental Price Prediction & Food Hamper Forecasting")

    # Subsection: SARIMA Model for Food Hampers
    st.subheader("Food Hamper Forecasting (SARIMA Model)")
    days = st.number_input("Enter the number of days to forecast:", min_value=1, step=1, value=1)

    if st.button("Predict Food Hampers"):
        predictions, fig = predict_for_days(int(days))
        if fig:
            st.pyplot(fig)
            total_hampers = sum(predictions)
            st.success(f"For {days} days, you will need approximately {int(total_hampers)} food hampers.")

    # Subsection: Kijiji Rental Price Prediction
    st.subheader("Kijiji Rental Price Prediction")
    property_type = st.selectbox("Type of Property", ['Apartment', 'House', 'Condo', 'Townhouse'])
    bedrooms = st.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 3, 1)
    size = st.slider("Size (sqft)", 300, 5000, 1000)
    unique_locations = data['CSDNAME'].unique()
    location = st.selectbox("Location", unique_locations)

    if st.button("Predict Rental Price"):
        # Load the trained model including preprocessing
        model = joblib.load('random_forest_regressor_model.pkl')

        # Prepare input data as a DataFrame
        input_df = pd.DataFrame({
            'Type': [property_type],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Size': [size],
            'CSDNAME': [location]
        })

        # Make prediction
        prediction = model.predict(input_df)

        # Map the predicted classes to labels
        price_bins = [0, 1700, 2300, float('inf')]
        price_labels = ['low', 'medium', 'high']
        price_category = pd.cut(prediction, bins=price_bins, labels=price_labels)

        # Display the predictions
        st.success(f"Predicted Rental Price: ${prediction[0]:,.2f}")
        st.success(f"Predicted Price Category: {price_category[0]}")

# Main App Logic
def main():
    st.sidebar.title("Kijiji Community App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()

if __name__ == "__main__":
    main()
