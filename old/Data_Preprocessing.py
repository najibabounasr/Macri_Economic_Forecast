#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing
# 
# ## Steps:
# 
# 1) Load the data, from our csv files. we will be using official data from the FED, and the BEA, aswell as BLS. 
# 2) We will be using nominal data, rather than real data initially, because we will then be applying the 'cpi-universal' to convert all data to real data.
# 3) We will then apply pct_change to the data to make it stationary.
# 4) We then properly label every column and we may further preprocess the data if necessary.

# In[187]:


import pandas as pd
import pandas as pd
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


# In[188]:


# Example Usage
base_path = '/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/CPI'  # Replace this with the actual base path
final_df = load_and_process_cpi_data(base_path)
print(final_df.head())  # Print the first few rows to verify the output

display(final_df)

cpi_data = final_df


# In[189]:


import pandas as pd


cpi_data = prepare_cpi_data(cpi_data)
display(cpi_data.head(5))


# # Switching Plans: Using more extensive data.
# 
# the date time range is too short, as it only measures from around 1999. We will need decades of data, and so we will be using a different dataset. 
# 

# In[190]:


def merge_interest_data( new_data, new_name, new_data_col_name):
    new_data['DATE'] = pd.to_datetime(new_data['DATE'])
    
    # Merge the dataframes on the 'Date' column
    merged_data = pd.merge(cpi_data, new_data, left_on='Date', right_on='DATE', how='left')
    
    # Drop the extra 'DATE' column from the new data
    merged_data.drop('DATE', axis=1, inplace=True)

    # Rename the new data column to 'Real GDP'
    merged_data.rename(columns={new_data_col_name: new_name}, inplace=True)
    return merged_data


# In[191]:


import pandas as pd

def load_and_rename_date_column(csv_filepath):
    # Load the data from a CSV file
    data = pd.read_csv(csv_filepath)
    
    # Rename the 'DATE' column to 'Date'
    data.rename(columns={'DATE': 'Date'}, inplace=True)
    
    return data

# Example usage
file_path = 'Data/Federal_Funds_Effective_Rate/FEDFUNDS.csv'
fed_funds_data = load_and_rename_date_column(file_path)

# Display the first few rows to verify
display(fed_funds_data.head())

cpi_data = fed_funds_data


# # GDP

# In[192]:


# Example usage:

# Your existing cpi_data is already prepared using prepare_cpi_data function

# New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/GDP/GDP/GDP.csv')
new_name = 'GDP'
new_data_col_name = 'GDP'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name, new_data_col_name)

# Display the first 20 rows to verify
display(cpi_data.head(20))


# # Universal Price Inflation: CPI

# In[193]:


# Example usage:

# Your existing cpi_data is already prepared using prepare_cpi_data function

# New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/CPI/CPI-Universal/CPIAUCSL.csv')
new_name = 'CPIAUCSL'
new_data_col_name = 'CPIAUCSL'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name, new_data_col_name)

# Display the first 20 rows to verify
display(cpi_data.head(20))


# # CPI- Shelter
# 

# In[ ]:


# Example usage:

# Your existing cpi_data is already prepared using prepare_cpi_data function

# New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/CPI/CPI-Housing/CUSR0000SAH1.csv')
new_name = 'CUSR0000SAH1'
new_data_col_name = 'CUSR0000SAH1'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name, new_data_col_name)

# Display the first 20 rows to verify
display(cpi_data.head(20))


# # Core Price Inflation:

# In[ ]:





# # Personal Consumption Expenditure
# 

# In[194]:


# New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Personal_Consumption/PCE.csv')
new_name = 'PCE'
new_data_col_name = 'PCE'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name, new_data_col_name)

display(cpi_data.head(20))


# # Private Residential Fixed Investment

# In[195]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Private_Residential_Fixed_Investment/private_residential_fixed_investment.csv')
new_name = 'PRFI'
new_data_col_name = 'PRFI'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Private Non-Residential Investment 
# 

# In[196]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Private_Nonresidential_Fixed_Investment/PNFI.csv')
new_name = 'PNFI'
new_data_col_name = 'PNFI'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Exports of Goods and Services

# In[197]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Exports_of_Goods_and_Services/EXPGS.csv')
new_name = 'EXPGS'
new_data_col_name = 'EXPGS'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Imports of Goods and Services
# 

# In[198]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Imports_of_Goods_and_Services/IMPGS.csv')
new_name = 'IMPGS'
new_data_col_name = 'IMPGS'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Government Consumption Expenditure
# 

# In[199]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Government_Consumption_Expenditures_and_Gross_Investment/GCE.csv')
new_name = 'GCE'
new_data_col_name = 'GCE'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))



# # Federal Consumption Expenditures and Gross Investment

# In[200]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Federal_Consumption_Expenditures_and_Gross_Investment/FGCE.csv')
new_name = 'FGCE'
new_data_col_name = 'FGCE'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Gross Domestic Product: Chain-type Price Index
# 
# 

# In[201]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Gross_Domestic_Product_Chain_type_Price_Index/GDPCTPI.csv')
new_name = 'GDPCTPI'
new_data_col_name = 'GDPCTPI'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Personal Consumption Expenditures: Chain Price Index

# In[202]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Personal_Consumption_Expenditures_Chain_type_Price_Index/PCEPI.csv')
new_name = 'PCEPI'
new_data_col_name = 'PCEPI'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)

# In[203]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Personal_Consumption_Expenditures_Excluding_Food_and_Energy_Chain_type_Price_Index/PCEPILFE.csv')
new_name = 'PCEPILFE'
new_data_col_name = 'PCEPILFE'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))



# # Business Sector: Hourly Compensation for All Workers

# In[204]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Business_Sector_Hourly_Compensation_for_All_Workers/PRS84006101.csv')
new_name = 'BSHCFAW'
new_data_col_name = 'PRS84006101'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # All Employees, Total Nonfarm

# In[205]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/All_Employees_Total_Nonfarm/PAYEMS.csv')
new_name = 'PAYEMS'
new_data_col_name = 'PAYEMS'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Civilian Unemployment Rate: 16 yr +

# In[206]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Unemployment_Rate/UNRATE.csv')
new_name = 'UNRATE'
new_data_col_name = 'UNRATE'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Industrial Production: Total Index

# In[207]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Industrial_Production_Total_Index/INDPRO.csv')
new_name = 'INDPRO'
new_data_col_name = 'INDPRO'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Capacity Utilization: Manufacturing [SIC]

# # 

# In[208]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Capacity_Utilization_Manufacturing/CUMFNS.csv')
new_name = 'CUMFNS'
new_data_col_name = 'CUMFNS'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Housing Starts: Total: New Privately Owned Housing Units Started

# #

# In[209]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Housing_Starts/HOUST.csv')
new_name = 'HOUST'
new_data_col_name = 'HOUST'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Disposable Personal Income
# 

# In[210]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Disposable_Personal_Income/DSPI.csv')
new_name = 'DSPI'
new_data_col_name = 'DSPI'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # University of Michigan Consumer Sentiment Index

# In[211]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/UMich_Consumer_Sentiment/tbmics.csv')
new_name = 'ICS_ALL'
new_data_col_name = 'ICS_ALL'

# Convert 'DATE' columns to datetime
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
new_data['Date'] = pd.to_datetime(new_data['Date'])
    
# Merge the dataframes on the 'Date' column
merged_data = pd.merge(cpi_data, new_data, left_on='Date', right_on='Date', how='left')

# Rename the new data column to 'Real GDP'
merged_data.rename(columns={new_data_col_name: new_name}, inplace=True)


# In[212]:


display(merged_data.head(10))

cpi_data = merged_data


# # Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity, Quoted on an Investment Basis

# In[213]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/2_Year_Treasury_Note_Yield_at_Constant_Maturity/DGS2.csv')
new_name = 'DGS2'
new_data_col_name = 'DGS2'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))



# # Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity, Quoted on an Investment Basis 

# # 

# In[214]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/5_Year_Treasury_Note_Yield/DGS5.csv')
new_name = 'DGS5'
new_data_col_name = 'DGS5'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed

# In[215]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/10_Year_Treasury_Note_Yield/DGS10.csv')
new_name = 'DGS10'
new_data_col_name = 'DGS10'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))



# # Moody's Seasoned Aaa Corporate Bond Yield

# In[216]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Moodys_Seasoned_Aaa_Corporate_Bond_Yield/AAA.csv')
new_name = 'AAA'
new_data_col_name = 'AAA'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))



# # Moody's Seasoned Baa Corporate Bond Yield

# In[217]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Moodys_Seasoned_Baa_Corporate_Bond_Yield/BAA.csv')
new_name = 'BAA'
new_data_col_name = 'BAA'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Nominal Broad US Dollar Index

# In[218]:


#New data reading from a CSV or Excel file
#Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Nominal_Broad_US_Dollar_Index/DTWEXBGS.csv')
new_name = 'DTWEXBGS'
new_data_col_name = 'DTWEXBGS'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# # Spot Crude Oil Price: West Texas Intermediate (WTI)

# In[219]:


#New data reading from a CSV or Excel file
# Assuming 'new_data.csv' is the file with the new data that has the 'DATE' and 'A191RL1Q225SBEA' columns
new_data = pd.read_csv('/Users/najibabounasr/Desktop/EJADA/Forecast Macroeconomic Trend/Data/Spot_Crude_Oil_Price_West_Texas_Intermediate/WTISPLC.csv')
new_name = 'WTISPLC'
new_data_col_name = 'WTISPLC'
# Merge the new data
cpi_data = merge_new_data(cpi_data, new_data, new_name,new_data_col_name)

display(cpi_data.head(10))


# In[ ]:





# # NEXT STEPS (if you forgot)
# # load all the data, then AT THE END apply the CPI univerrsal to turn all nominal data into real data. after that we can turn everything into pct_change data, before we begin model training. 

# # Step 2: Process the Data. 

# In[220]:


merged_dataframe = cpi_data
display(merged_dataframe.head(10))


# # Step 1:
# # Turning Nominal Values into Real Values
# 
# Here, we adjust the values of our data to reflect real values, rather than nominal values. This is done by applying the 'cpi-universal' to our data.

# In[221]:


def deflate_nominal_values(df, cpi_col_name, columns_to_deflate):
    """ Deflate nominal data using the CPI index to real values. """
    for col in columns_to_deflate:
        df[col] = df[col] / df[cpi_col_name] * 100  # Use the column name directly
    return df

# Correct usage:
cpi_col_name = 'CPI-Universal' 
columns_to_deflate = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'DSPI']

# Now apply the function using the corrected parameter
deflated_df = deflate_nominal_values(merged_dataframe, cpi_col_name, columns_to_deflate)

# Display the first few rows to check the result
display(deflated_df.head(10))


# # Step 2: 
# # Reducing Variance: Applying logarithmic transformation to the data

# In[ ]:


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
display(log_transformed_df.head(10))


# # Missing Data: (ADD LATER)
# 
# ## Stock Price Index: Standard & Poor’s 500 Composite: 
# ## S&P GSCI Non-Energy Commodities Nearby Index: 
# ## S&P 500 VOLATILITY INDEX: VIX

# In[ ]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd
from funcs.machine_learning import check_stationarity, plot_series_stationarity


# In[ ]:


# Assuming 'df' is your DataFrame
log_transformed_df['Date'] = pd.to_datetime(log_transformed_df['Date'])
log_transformed_df.set_index('Date', inplace=True)


# In[ ]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd



# # Example usage of functions:

# In[ ]:


# Assume 'df' is your DataFrame with 'Date' as index
series = log_transformed_df['InterestRateColumn']  # replace with the actual column name for interest rates

# Check stationarity
check_stationarity(series)

# Plot the stationarity
plot_series_stationarity(series, window=12)

