import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.special import inv_boxcox

# Function to plot SARIMA Box-Cox Transformed Graph
def plot_boxcox_forecast(train_df, test_df, forecast_values_boxcox, confidence_intervals_boxcox):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        train_df['date'],
        train_df['actual_pickup_boxcox'],
        label='Actual Pickups (Train) - Box-Cox Transformed',
        color='blue'
    )

    ax.plot(
        test_df['date'],
        test_df['actual_pickup_boxcox'],
        label='Actual Pickups (Test) - Box-Cox Transformed',
        color='orange'
    )

    ax.plot(
        test_df['date'],
        forecast_values_boxcox,
        label='Forecasted Pickups (Box-Cox Transformed)',
        color='green'
    )

    ax.fill_between(
        test_df['date'],
        confidence_intervals_boxcox.iloc[:, 0],
        confidence_intervals_boxcox.iloc[:, 1],
        color='gray',
        alpha=0.3,
        label='Confidence Interval'
    )

    ax.set_title('SARIMA Forecast for Box-Cox Transformed Pickups (Train vs Test)')
    ax.legend()
    ax.grid(True)

    return fig

# Function to plot SARIMA Original Scale Graph
def plot_original_scale_forecast(df, test_df, forecast_values_original):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df['date'],
        df['actual_pickup'],
        label='Actual Pickups',
        color='blue'
    )

    ax.plot(
        test_df['date'],
        forecast_values_original,
        label='Forecasted Pickups (Reversed)',
        color='green'
    )

    ax.set_title('SARIMA Forecast for Pickups (Original Scale, Reversed Box-Cox)')
    ax.legend()
    ax.grid(True)

    return fig

# Streamlit app for displaying SARIMA graphs
def display_sarima_graphs(train_df, test_df, df, forecast_values_boxcox, confidence_intervals_boxcox, forecast_values_original):
    st.title("SARIMA Forecast Analysis")

    # Add first graph: Box-Cox Transformed
    st.subheader("Box-Cox Transformed Forecast")
    fig1 = plot_boxcox_forecast(train_df, test_df, forecast_values_boxcox, confidence_intervals_boxcox)
    st.pyplot(fig1)

    # Add second graph: Reversed Transformation to Original Scale
    st.subheader("Forecast on Original Scale")
    fig2 = plot_original_scale_forecast(df, test_df, forecast_values_original)
    st.pyplot(fig2)

# Example call within Streamlit
def main():
    # Simulate data and forecasts for demonstration purposes
    # In real implementation, replace with actual data and model results

    # Assuming the DataFrame `df` and model forecasts are already computed
    df = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=100),
        "actual_pickup": np.random.randint(50, 200, size=100),
        "actual_pickup_boxcox": np.random.random(size=100)
    })

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    forecast_values_boxcox = np.random.random(size=len(test_df))
    confidence_intervals_boxcox = pd.DataFrame({
        0: forecast_values_boxcox - 0.1,  # Lower bound
        1: forecast_values_boxcox + 0.1   # Upper bound
    })
    forecast_values_original = np.random.randint(50, 200, size=len(test_df))

    display_sarima_graphs(train_df, test_df, df, forecast_values_boxcox, confidence_intervals_boxcox, forecast_values_original)

if __name__ == "__main__":
    main()
