import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
######-------- FUNCTIONS -------------#####################################################

# 1. Evaluate each transformation and find the best based on ADF statistic
# Evaluate each transformation and find the best based on ADF statistic
def evaluate_transformations(data):
    methods = {
        'None': data,
        'Simple Differencing': data.diff().dropna(),
        'Rolling Mean Subtraction': (data - data.rolling(window=7).mean()).dropna(),
        'Rolling Mean Subtraction + Differencing': (data - data.rolling(window=7).mean()).diff().dropna()
    }

    results = {}
    for method, transformed_data in methods.items():
        adf_result = adfuller(transformed_data)
        results[method] = adf_result[0]  # Storing the ADF statistic

    # Find the method with the smallest ADF statistic
    best_method = min(results, key=results.get)
    best_adf_statistic = results[best_method]

    return best_method, best_adf_statistic

##########################################################################################

# Title and introduction
st.title("Macroeconomic Forecasting: ARIMA Model")
st.write("An interactive tool for forecasting economic indicators using the ARIMA model.")

# Data loading
@st.cache
def load_data():
    df = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macro Economic Trend/streamlit_ready_data.csv', parse_dates=['Date'], index_col='Date')
    return df

df = load_data()

# Part 1: Data Selection and Visualization
with st.expander("Part 1: Data Selection and Visualization", expanded=True):
    option = st.selectbox('Select a series to forecast', df.columns)
    data = df[option]

    st.write("Raw Data Preview:")
    st.write(data.head())
    st.write("Summary Statistics:")
    st.write(data.describe())
    st.line_chart(data)

# Part 2: Data Analysis and Forecasting
with st.expander("Part 2: Data Analysis and Forecasting", expanded=False):
    st.subheader("Data Transformation for Stationarity")

    # Selection for the differencing method
    diff_option = st.selectbox("Select differencing method for visualization:",
                               ['None', 'Simple Differencing', 'Rolling Mean Subtraction', 'Rolling Mean Subtraction + Differencing'],
                               index=0)

    # Apply the selected transformation
    if diff_option == 'Simple Differencing':
        data_diff = data.diff().dropna()
    elif diff_option == 'Rolling Mean Subtraction':
        rolling_mean = data.rolling(window=7).mean()
        data_diff = (data - rolling_mean).dropna()
    elif diff_option == 'Rolling Mean Subtraction + Differencing':
        rolling_mean = data.rolling(window=7).mean()
        data_diff = (data - rolling_mean).diff().dropna()
    else:
        data_diff = data

    # ADF test and ACF, PACF plot visualization
    st.subheader("Augmented Dickey-Fuller Test and ACF, PACF Plots")
    if st.button("Perform ADF Test and Plot ACF, PACF"):
        # Perform ADF test
        result = adfuller(data_diff)
        st.write('ADF Statistic: %f' % result[0])
        st.write('p-value: %f' % result[1])
        st.write('Critical Values:')
        for key, value in result[4].items():
            st.write(f'{key}: {value:.3f}')

        # Create plots for ACF and PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        plot_acf(data_diff, lags=50, zero=False, ax=ax1)
        ax1.set_title('Autocorrelation Function')

        plot_pacf(data_diff, lags=50, zero=False, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function')

        st.pyplot(fig)
    st.markdown("---")
    st.warning("NOTE: The best transformation method will automatically be shown below.")
    st.subheader("Best Transformation Result:")
    best_method, best_statistic = evaluate_transformations(data)
    column_name = option
    st.write(f"The best transformation method for {column_name} is {best_method} with an ADF statistic of {best_statistic:.2f}")

# Part 3: Table of Best Transformation Results for All Series
with st.expander("Part 3: Table of Best Transformation Results for All Series", expanded=False):
    results = {}
    for column in df.columns:
        best_method, best_statistic = evaluate_transformations(df[column])
        results[column] = [best_method, best_statistic]

    results_df = pd.DataFrame(results, index=['Best Method', 'ADF Statistic']).T
    st.write(results_df)
    # Display a bar plot of the ADF statistics using plotly
    st.bar_chart(results_df['ADF Statistic'])
    # Save the results to a CSV file
    save_path = "/Users/najibabounasr/Desktop/EJADA/Forecast Macro Economic Trend/Testing/Results/ARIMA/ARIMA_best_transformations.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=True)
    st.write(f"Results saved to: {save_path}")

# Part 4: ARIMA Model Fitting and Forecasting