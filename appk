import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.special import inv_boxcox
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
# Set up the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', st.secrets.get("GOOGLE_API_KEY"))
genai.configure(api_key=GOOGLE_API_KEY)
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to generate response from the model
def generate_response(prompt, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        # Include context from uploaded data in the prompt
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text  # Use 'text' attribute
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."


# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')
# Load the dataset with a specified encoding
data = pd.read_csv('dataframe.csv', encoding='latin1')
def load_data_and_model():
    # Load your dataset (replace 'your_dataset.csv')
    df = pd.read_csv('dataframe.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Define your SARIMA model parameters
    sarima_model = sm.tsa.SARIMAX(
        df['actual_pickup_boxcox'],
        exog=df[['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']],
        order=(4, 1, 4),
        seasonal_order=(1, 1, 1, 7)
    )

    sarima_fit = sarima_model.fit(disp=False)
    return df, sarima_fit

df, sarima_fit = load_data_and_model()# Streamlit application
# Page 1: Dashboard
def dashboard():
        # Add an image
    st.image("downloads.png", use_column_width=True)

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
def machine_learning_modeling():
    st.title("Food Hamper Pickup Predictor")
    st.write("Enter the date range to predict food hamper pickups and visualize the results.")

machine_learning_modeling()

# User Input: Start and End Dates
start_date = st.text_input("Start Date (YYYY-MM-DD):", "2024-01-01")
end_date = st.text_input("End Date (YYYY-MM-DD):", "2024-01-15")

if st.button("Predict"):
    try:
        # Convert input dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate prediction dates
        prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Prepare exogenous variables for prediction
        exog_future = df.loc[df['date'].isin(prediction_dates), 
                             ['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']]
        if exog_future.empty:
            st.error("No data available for the specified dates. Please choose another range.")
        else:
            # Forecast with SARIMA model
            forecast = sarima_fit.get_forecast(steps=len(exog_future), exog=exog_future)
            forecast_values_boxcox = forecast.predicted_mean

            # Reverse Box-Cox Transformation
            forecast_values_original = inv_boxcox(forecast_values_boxcox, sarima_fit.params['lambda'])

            # Create a DataFrame for predictions
            prediction_df = pd.DataFrame({
                'date': prediction_dates,
                'predicted_pickups': forecast_values_original
            })

            # Display predictions
            st.subheader("Predicted Food Hamper Pickups")
            st.dataframe(prediction_df)

            # Plot predictions
            st.subheader("Prediction Graph")
            plt.figure(figsize=(10, 6))
            plt.plot(prediction_df['date'], prediction_df['predicted_pickups'], marker='o', label='Predicted Pickups')
            plt.title('Predicted Food Hamper Pickups')
            plt.xlabel('Date')
            plt.ylabel('Number of Pickups')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Page 4: Display SARIMA Forecast Graphs
# def sarima_forecast_graphs():
#     st.title("SARIMA Forecast Graphs")

#     # Convert 'date' column to datetime
#     data['date'] = pd.to_datetime(data['date'])

#     # Prepare train and test sets
#     train_df = data.iloc[:int(len(data) * 0.8)]  # First 80% of the data
#     test_df = data.iloc[int(len(data) * 0.8):]  # Last 20% of the data

#     # Generate exogenous variables for test set
#     future_exog = test_df[['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']]

#     # Forecast using SARIMA model
#     forecast_values_boxcox = sarima_model.forecast(steps=len(test_df), exog=future_exog)
#     confidence_intervals_boxcox = pd.DataFrame({
#         0: forecast_values_boxcox - 0.1,  # Placeholder lower bound (adjust with actual)
#         1: forecast_values_boxcox + 0.1   # Placeholder upper bound (adjust with actual)
#     })

#     # Reverse Box-Cox transformation (placeholder for actual lambda)
#     forecast_values_original = np.exp(forecast_values_boxcox)  # Adjust transformation as needed

#     # Plot the Box-Cox transformed graph
#     st.subheader("Forecast (Box-Cox Transformed Data)")
#     fig1 = plot_boxcox_graph(train_df, test_df, forecast_values_boxcox, confidence_intervals_boxcox)
#     st.pyplot(fig1)

#     # Plot the reversed transformation graph
#     st.subheader("Forecast (Original Scale)")
#     fig2 = plot_original_graph(data, test_df, forecast_values_original)
#     st.pyplot(fig2)
# Page 5: Map
def map():
    st.title("Map for Food Hamper Prediction.")
    st.markdown("""<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1Uf7Agld8GzoH9-fzNNsUpmCN-0X8BEQ&ehbc=2E312F" width="640" height="480"></iframe>
    """, unsafe_allow_html=True)
#Page 6
# Streamlit app
def chatbot():
    st.title("Food Hamper Distribution Chatbot")
    st.write("Reading files from predefined paths...")

    # List of predefined file paths
    file_paths = ["Final progress report1 .pdf", "mergedfoodandclientfinal.xlsx"
    ]

    # Prepare data context
    data_context = ""
    for file_path in file_paths:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data_context += f"\nData from {file_path}:\n{df.head(5).to_string()}\n"
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                data_context += f"\nData from {file_path}:\n{df.head(5).to_string()}\n"
            elif file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
                data_context += f"\nExtracted text from {file_path}:\n{text[:1000]}...\n"  # Limit to first 1000 characters
            st.success(f"Successfully processed {file_path}")
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your project:", key="input")
    if st.button("Send"):
        if user_input and data_context:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input, data_context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif not data_context:
            st.error("No valid data context available. Please check the file paths.")

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction")
    app_page = st.sidebar.radio(
        "Select a Page",
        ["Dashboard", "Data visualizations", "Sarima Model Predictions", "Map for Food Hamper Prediction", "Chatbot"]
    )

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Data visualizations":
        exploratory_data_analysis()
    elif app_page == "Sarima Model Predictions":
        machine_learning_modeling()
    # elif app_page == "SARIMA Forecast Graphs":
    #     sarima_forecast_graphs()
    elif app_page == "Map for Food Hamper Prediction":
        map()
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
