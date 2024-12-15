import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Function to train the SARIMA model and forecast future values
def train_and_forecast_sarima(df, future_steps, lam):
    """
    Train a SARIMA model on historical data and forecast future values.

    Parameters:
        df (DataFrame): Input DataFrame containing historical data.
        future_steps (int): Number of future time steps to predict.
        lam (float): Box-Cox transformation parameter.

    Returns:
        forecast_values_original (ndarray): Forecasted values on the original scale.
        confidence_intervals_original (DataFrame): Confidence intervals for the forecasts on the original scale.
        forecast_df (DataFrame): DataFrame containing forecasted values and confidence intervals.
        model_fit (SARIMAXResults): Trained SARIMA model fit object.
    """
    # Ensure the data is sorted by date
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Define training data
    train_y = df['actual_pickup_boxcox']
    exog_train = df[['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']]

    # Train SARIMA model
    sarima_model = sm.tsa.SARIMAX(
        train_y,
        exog=exog_train,
        order=(4, 1, 4),  # Tune these parameters as needed
        seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = sarima_model.fit(disp=False)

    # Print model summary
    print(model_fit.summary())

    # Create future exogenous variables for prediction
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
    future_exog = {
        "scheduled_pickup": [100 + i * 2 for i in range(future_steps)],
        "scheduled_pickup_lag_7": [90 + i for i in range(future_steps)],
        "scheduled_pickup_lag_14": [80 + i for i in range(future_steps)],
    }
    future_exog_df = pd.DataFrame(future_exog, index=future_dates)

    # Forecast future values
    forecast_results = model_fit.get_forecast(steps=future_steps, exog=future_exog_df)
    forecast_values_boxcox = forecast_results.predicted_mean
    confidence_intervals_boxcox = forecast_results.conf_int()

    # Inverse Box-Cox transformation to return values to the original scale
    forecast_values_original = inv_boxcox(forecast_values_boxcox, lam)
    confidence_intervals_original = inv_boxcox(confidence_intervals_boxcox, lam)

    # Combine forecasts and confidence intervals into a DataFrame
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Values": forecast_values_original,
        "Lower CI": confidence_intervals_original[:, 0],
        "Upper CI": confidence_intervals_original[:, 1],
    })

    return forecast_values_original, confidence_intervals_original, forecast_df, model_fit

# Visualize the forecast
def plot_forecast(forecast_df):
    """
    Plot the forecasted values with confidence intervals.

    Parameters:
        forecast_df (DataFrame): DataFrame containing forecasted values and confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['Date'], forecast_df['Forecasted Values'], label='Forecasted Values', marker='o')
    plt.fill_between(
        forecast_df['Date'],
        forecast_df['Lower CI'],
        forecast_df['Upper CI'],
        color='gray',
        alpha=0.3,
        label='Confidence Interval'
    )
    plt.title("Future Forecast of Food Hampers (SARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Number of Food Hampers")
    plt.legend()
    plt.grid()
    plt.show()

# Main logic
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("dataframe.csv", encoding="latin1")

    # Parameters
    future_steps = 30  # Number of future days to forecast
    lam = 0.5  # Box-Cox transformation parameter; adjust based on training data

    # Train and forecast using SARIMA
    forecast_values, confidence_intervals, forecast_df, model_fit = train_and_forecast_sarima(df, future_steps, lam)

    # Display forecasted results
    print("Forecasted Values:")
    print(forecast_df)

    # Plot the forecast
    plot_forecast(forecast_df)
