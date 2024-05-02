#!/usr/bin/env python
# coding: utf-8

# # Data Download and Preprocessing
# 
# ## Steps:
# 
# 1) Load the data, from our csv files. we will be using official data from the FED, and the BEA, aswell as BLS. 
# 2) We will be using nominal data, rather than real data initially, because we will then be applying the 'cpi-universal' to convert all data to real data.
# 3) We will then apply pct_change to the data to make it stationary.
# 4) We then properly label every column and we may further preprocess the data if necessary.

# In[16]:


import pandas as pd
import pandas as pd
get_ipython().system(' pip install fredapi')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from funcs import loading_csv_functions
from funcs.loading_csv_functions import merge_new_data, merge_new_data_and_apply_pct_change, prepare_cpi_data, preprocess_and_merge
from funcs.loading_csv_functions import load_and_process_cpi_data
def name(self) -> any:
    return self.attribute

import os


# In[17]:


import pandas as pd
from local_settings import settings
from fredapi import Fred
import requests

# Use the settings dictionary
api_key = settings['api_key']
series_ids = settings['series_ids']
start_date = settings['start_date']
end_date = settings['end_date']
# Base URL for API requests
base_url = 'https://api.stlouisfed.org/fred/series/observations'

# Initialize the FRED API with your API key
fred = Fred(api_key=settings['api_key'])


# In[18]:


import pandas as pd
from fredapi import Fred
from local_settings import settings  # Ensure this import correctly brings in your settings

# Initialize the FRED API with the API key from settings
fred = Fred(api_key=settings['api_key'])

# Function to fetch and prepare data
def fetch_data(series_id):
    try:
        print(f"Fetching data for {series_id}")
        data = fred.get_series(series_id, observation_start=settings['start_date'], observation_end=settings['end_date'])
        data.index = pd.to_datetime(data.index)  # Convert index to datetime
        return pd.DataFrame(data, columns=[series_id])
    except Exception as e:
        print(f"Error fetching data for {series_id}: {str(e)}")
        return pd.DataFrame()

# Fetch and store data for each series in a dictionary
data_frames = {}
for series_id in settings['series_ids']:
    data_frame = fetch_data(series_id)
    if not data_frame.empty:
        data_frames[series_id] = data_frame

# Combine all data into a single DataFrame
combined_data = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())

# Display the combined DataFrame
print(combined_data.tail())


# In[19]:


# drop level, to get rid of double headers
combined_data.columns = combined_data.columns.droplevel()


# In[20]:


# Show the combined data
display(combined_data.tail())


# # Display the monthly data

display(monthly_data.tail())


# # First stationarity check

# # OLD SCRIPT:
# 
# 
# ---

# # Part 2: Process the Data. 

# # Step 1:
# # Turning Nominal Values into Real Values
# 
# Here, we adjust the values of our data to reflect real values, rather than nominal values. This is done by applying the 'cpi-universal' to our data.

# In[21]:


def deflate_nominal_values(df, cpi_col_name, columns_to_deflate):
    """ Deflate nominal data using the CPI index to real values. """
    for col in columns_to_deflate:
        df[col] = df[col] / df[cpi_col_name] * 100  # Use the column name directly
    return df

# Correct usage:
cpi_col_name = 'CPIAUCSL' 
columns_to_deflate = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'DSPI']

# Now apply the function using the corrected parameter
deflated_df = deflate_nominal_values(combined_data, cpi_col_name, columns_to_deflate)

# Display the first few rows to check the result
display(deflated_df.head())


# # Step 2: 
# # Reducing Variance: Applying logarithmic transformation to the data

# In[22]:


import numpy as np
import pandas as pd

def apply_log_transformations(df, columns_to_transform):
    """ Apply the 100 * log transformation to specified DataFrame columns. """
    for col in columns_to_transform:
        df[col] = 100 * np.log(df[col])
    return df

# Example usage:
# Define the columns that need the logarithmic transformation
columns_to_transform = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'HOUST', 'DSPI']

# Assuming 'deflated_df' is your DataFrame after deflation and is now ready for transformations
log_transformed_df = apply_log_transformations(deflated_df, columns_to_transform)

# Display the first few rows to check the result
display(log_transformed_df.head())


# In[26]:


import pandas as pd
df = log_transformed_df 
# Assuming 'df' is your DataFrame and the index is datetime
df.index = pd.to_datetime(df.index)  # Ensure the index is in datetime format

# Filter data to only include the first day of each month
monthly_data = df[df.index.is_month_start]

# Display the resulting DataFrame
display(monthly_data)

# save as csv 'economic_data.csv
df = monthly_data
df.to_csv('economic_data.csv')


# # Missing Data: (ADD LATER)
# 
# ## Stock Price Index: Standard & Poor’s 500 Composite: 
# ## S&P GSCI Non-Energy Commodities Nearby Index: 
# ## S&P 500 VOLATILITY INDEX: VIX
