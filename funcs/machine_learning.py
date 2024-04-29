import pandas as pd
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
# Define a function to perform the Augmented Dickey-Fuller test
def check_stationarity(data):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity.
    
    Arguments:
    Pandas Series: a series of data to be checked for stationarity.
    
    Returns:
    Prints test statistics and critical values.
    """
    # Perform Augmented Dickey-Fuller test
    # Perform the test using the AIC criterion for choosing the number of lags
    print('Results of Augmented Dickey-Fuller Test:')
    adf_test = adfuller(data, autolag='AIC')  

    # Extract and print the test statistics and critical values
    adf_output = pd.Series(adf_test[0:4], 
                           index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)
    return adf_output


import matplotlib.pyplot as plt

def plot_series_stationarity(series, window=12):
    """
    Plot the time series, its rolling mean, and its rolling standard deviation.
    
    Arguments:
    series: Pandas Series - the time series to plot.
    window: int - the window size for calculating rolling statistics.
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plot the statistics
    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Original Series')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std Dev')
    plt.title('Time Series Stationarity Check')
    plt.legend()
    plt.show()