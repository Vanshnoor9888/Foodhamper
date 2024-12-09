import streamlit as st
import pandas as pd
import joblib
import folium
# Load the dataset with a specified encoding
data = pd.read_csv('mergedfoodandclients.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.subheader("💡 Abstract:")
    inspiration = '''
Data Quality: It is impossible to exaggerate the significance of data quality. An essential first step in guaranteeing the precision and dependability of our analysis and models was cleaning and preparing the dataset.
Feature Selection: The effectiveness of machine learning models is greatly impacted by the identification of pertinent features. We identified the key variables influencing Ontario rental pricing through iterative experimentation.
Model Evaluation: To appropriately determine a machine learning model's performance and capacity for generalization, a thorough evaluation of the model is necessary. We assessed and improved our models using a range of metrics and methods.
Deployment Obstacles: Scalability, security, and system integration are just a few of the difficulties that come with deploying machine learning models to commercial settings. Working together across several teams and areas of expertise was necessary to address these problems.
Overall, this study offered insightful information about the rental market in Ontario and the practical uses of machine learning methods. It emphasized how crucial it is for data science projects to have interdisciplinary collaboration and ongoing learning.
    '''
    st.write(inspiration)
    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
  The purpose of this research is to use machine learning techniques to perform an extensive examination of the rental market in Ontario, Canada. The project will be broken down into three primary stages: the creation of machine learning (ML) models, deployment, and exploratory data analysis (EDA) and visualization.In order to obtain insights into the trends, patterns, and factors impacting rental pricing in the rental market, a range of statistical approaches and visualization tools will be utilized throughout the EDA phase. In this stage, the rental data will be cleaned and preprocessed, outliers and missing values will be found, and correlations between various factors will be investigated.Using supervised learning methods like regression and classification, predictive models will be constructed throughout the machine learning model building phase in order to forecast rental prices and examine the variables influencing price fluctuations.
  Furthermore, rental market segmentation based on various attributes may be achieved through the use of unsupervised learning techniques such as clustering.In the Deployment phase, the built machine learning models will be made available to customers via a web platform or application. This will enable them to interactively explore insights about the rental market and receive rental price projections based on predetermined criteria.
     '''



# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Price Distribution
    # fig = px.scatter(data, x='Size', y='Price', trendline="ols", title='Relationship between Size and Price')
    # st.plotly_chart(fig)
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/b91808fe-0100-4e7f-94d4-957c4fea0c20/page/AtrGE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)
# Page 3: Machine Learning Modeling
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
Data Quality: It is impossible to exaggerate the significance of data quality. An essential first step in guaranteeing the precision and dependability of our analysis and models was cleaning and preparing the dataset.
Feature Selection: The effectiveness of machine learning models is greatly impacted by the identification of pertinent features. We identified the key variables influencing Ontario rental pricing through iterative experimentation.
Model Evaluation: To appropriately determine a machine learning model's performance and capacity for generalization, a thorough evaluation of the model is necessary. We assessed and improved our models using a range of metrics and methods.
Deployment Obstacles: Scalability, security, and system integration are just a few of the difficulties that come with deploying machine learning models to commercial settings. Working together across several teams and areas of expertise was necessary to address these problems.
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
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Community Mapping"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()

if __name__ ==+ "__main__":
    main()
