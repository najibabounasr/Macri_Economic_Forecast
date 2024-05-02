#!/usr/bin/env python
# coding: utf-8

# # Beggining Training
# ### We will begin training our model by importing the necessary libraries and loading the data.

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# we will be creating a VAR model, install the statsmodels library
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from funcs.machine_learning import check_stationarity, plot_series_stationarity

data = pd.read_csv('economic_data.csv', index_col='Date', parse_dates=True)



# In[34]:


# temporarily drop all columns with ANY missing values
# Drop all columns with any missing values
data = data.dropna(axis=1)


# In[35]:


for column in data.columns:
    print(f'Checking stationarity of {column}')
    check_stationarity(data[column])
    plot_series_stationarity(data[column])


# # Initial Test Results: 
# 
# To analyze the stationarity of the data, we can look at the p-values from the Augmented Dickey-Fuller (ADF) test. The null hypothesis of the ADF test is that the data has a unit root, implying it is non-stationary. Therefore, if the p-value is less than a chosen significance level (commonly 0.05), we reject the null hypothesis and consider the data stationary.
# 
# ### Stationary:
# FEDFUNDS: p-value = 0.045640 < 0.05
# 
# HOUST: p-value = 0.016689 < 0.05
# 
# UNRATE: p-value = 0.011867 < 0.05
# 
# CUMFNS: p-value = 0.003216 < 0.05
# 
# ### Non-Stationary:
# GDP: p-value = 0.207172 > 0.05
# 
# CPIAUCSL: p-value = 0.998625 > 0.05
# 
# CUSR0000SAH1: p-value = 0.998984 > 0.05
# 
# CPILFESL: p-value = 0.998666 > 0.05
# 
# PCE: p-value = 0.142191 > 0.05
# 
# PRFI: p-value = 0.475664 > 0.05
# 
# PNFI: p-value = 0.408358 > 0.05
# 
# EXPGS: p-value = 0.356946 > 0.05
# 
# DSPI: p-value = 0.126232 > 0.05
# 
# AAA: p-value = 0.518264 > 0.05
# 
# BAA: p-value = 0.460815 > 0.05
# 
# WTISPLC: p-value = 0.424639 > 0.05
# 
# IMPGS: p-value = 0.231460 > 0.05
# 
# GCE: p-value = 0.340913 > 0.05
# 
# FGCE: p-value = 0.573069 > 0.05
# 
# GDPCTPI: p-value = 0.998564 > 0.05
# 
# PCEPI: p-value = 0.997000 > 0.05
# 
# PCEPILFE: p-value = 0.995870 > 0.05
# 
# PAYEMS: p-value = 0.878303 > 0.05
# 
# INDPRO: p-value = 0.722095 > 0.05
# 

# # Stationarity Analysis
# ### Stationary Variables:
# Variables are typically considered stationary if the p-value is less than 0.05, which suggests that we can reject the null hypothesis that there is a unit root present in the time series. From your results, the variables considered stationary are:
# 
# HOUST: Housing starts, p-value = 0.016689.
# 
# UNRATE: Unemployment rate, p-value = 0.011867.
# 
# CUMFNS: Cumulative function, p-value = 0.003216.
# 
# ## Non-Stationary Variables:
# Variables with a p-value greater than 0.05 are generally considered non-stationary. This includes most of your variables, such as GDP, CPIAUCSL, FEDFUNDS, and others. The non-stationary variables need some form of differencing or transformation to achieve stationarity.
# 
# ## Insights and Recommendations
# #### High p-values for Price Indices: Variables like CPIAUCSL, PCEPI, and CUSR0000SAH1 have very high p-values close to 1. This indicates a very strong presence of a unit root, suggesting these series are highly non-stationary. Such series are typically persistent over time, reflecting cumulative inflation effects.
# 
# #### FEDFUNDS Close to Stationarity: The Federal Funds Rate (FEDFUNDS) has a p-value of 0.045640, which is marginally below the 0.05 threshold. This might suggest some level of stationarity, although it's close to the cutoff. This could be due to policy interventions that stabilize interest rates over certain periods.
# 
# #### Employment and Industrial Production: Variables related to economic activity like PAYEMS and INDPRO are not stationary, which might indicate cyclical patterns associated with business cycles.

# In[36]:


# Check initial NaN counts
print(data.isnull().sum())

# Optionally, fill NaNs or drop them
# data.fillna(method='ffill', inplace=True)  # Forward fill
# data.dropna(inplace=True)  # Drop rows with NaNs


# In[37]:


# Check for zeros
print((data == 0).sum())

# Apply pct_change and immediately handle infinities
data = data.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


# In[39]:


from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

def plot_series_stationarity(series):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title('Time Series (Stationarity Check)')
    plt.show()

# Display null counts after transformation
print(data.isnull().sum())

# Check and plot stationarity for each column
for column in data.columns:
    print(f'Checking stationarity of {column}')
    check_stationarity(data[column])
    plot_series_stationarity(data[column])

# Cut the dataframe so that it starts at 1976-02-01
data = data['1976-02-01':]

# save this data as a new 'streamlit_ready_data.csv' file
data.to_csv('streamlit_ready_data.csv')


# ## Variables that Became Stationary
# These variables have shown significant improvement in terms of stationarity as their p-values are now below 0.05, suggesting rejection of the null hypothesis of a unit root (indicating stationarity):
# 
# FEDFUNDS: Improved significantly, showing signs of stationarity.
# 
# GDP: Major improvement, indicating stationarity.
# 
# CPIAUCSL: Now shows stationarity which is a notable change.
# 
# CUSR0000SAH1, CPILFESL, PCE, PRFI, PNFI, EXPGS: All these variables have shown improvements and are now stationary.
# 
# HOUST: Remains stationary, with an even more significant p-value.
# 
# PCEPI: Shows substantial improvement and is now stationary.
# 
# ## Variables Still Non-Stationary
# These variables still show p-values above 0.05, indicating that they are not stationary and may require further differencing or transformations:
# 
# DSPI, AAA, BAA, WTISPLC: While some improvement is noted, these remain non-stationary.
# 
# IMPGS, GCE, FGCE, GDPCTPI: These continue to show non-stationarity and might need additional transformations or differencing.
# 
# PCEPILFE: Shows improvement but still hovers around the threshold of stationarity.
# 
# 

# In[ ]:


# import the Var module
from statsmodels.tsa.api import VAR 
# 1 Arima
# 2 SARIMA
# 3 FB Prophet
# 4 Regressor model
# 5 Random Forest model
# 6 lightgbm
# 7 XGboost
# 8 Auto gluon 

# ML-Flow  

