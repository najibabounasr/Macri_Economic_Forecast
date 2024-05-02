#!/usr/bin/env python
# coding: utf-8

# In[306]:


# Import the log_transformed_df.csv file from the data folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from funcs.machine_learning import check_stationarity, plot_series_stationarity
import itertools
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hvplot.pandas  # Import HvPlot for Pandas
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import dim, opts
from bokeh.plotting import show  # Import show function from Bokeh
from statsmodels.tsa.arima.model import ARIMA
import pickle


log_transformed_df = pd.read_csv('economic_data.csv')


# In[307]:


# First thing first: let's look at the entire top 50 and bottom 50 rows of the dataframe

display(log_transformed_df['PCE'].head(100))
display(log_transformed_df.tail(50))
# Change the name of the Unnamed: 0 column to 'Date' and set the column as the index
log_transformed_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
log_transformed_df.set_index('Date', inplace=True)


# In[308]:


# Assume 'df' is your DataFrame with 'Date' as index
series = log_transformed_df['FEDFUNDS']  # replace with the actual column name for interest rates

# Check stationarity
check_stationarity(series)

# Plot the stationarity
plot_series_stationarity(series, window=12)


# # Check how any NaN values are in the dataset

# In[309]:


# Check how many Nan values there are in the dataset

print(log_transformed_df.isnull().sum())


# Create a function that finds the columns that have NaN values
def find_Nan_columns(df):
    Nan_columns = []
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            Nan_columns.append(column)
    return Nan_columns


# Find the columns that have NaN values
Nan_columns = find_Nan_columns(log_transformed_df)
display(Nan_columns)
#print(Nan_columns)

quarterly_data = []
# Save all the columns that have 559 NaN values in a list of quarterly data
for Nan_columns in log_transformed_df.columns:
    if log_transformed_df[Nan_columns].isnull().sum() == 559:
        quarterly_data.append(Nan_columns)
    
display(quarterly_data) 
        


# In[310]:


# search for any 'string' values in the dataset
for column in log_transformed_df.columns:
    if log_transformed_df[column].dtype == 'object':
        print(column)
        
# forcefully convert the 'DGS5', and 'DGS10' columns to float
log_transformed_df['DGS5'] = pd.to_numeric(log_transformed_df['DGS5'], errors='coerce')
log_transformed_df['DGS10'] = pd.to_numeric(log_transformed_df['DGS10'], errors='coerce')
log_transformed_df['DGS2'] = pd.to_numeric(log_transformed_df['DGS2'], errors='coerce')
log_transformed_df['DTWEXBGS'] = pd.to_numeric(log_transformed_df['DTWEXBGS'], errors='coerce')

# Check if conversion was successful
display((log_transformed_df.dtypes))


# In[ ]:


# check for the data type of the columns
log_transformed_df.dtypes


# In[ ]:


# Find the index column name
index_column = log_transformed_df.index.name
display(index_column)

# convery 'date' to datetime if it is not already
log_transformed_df.index = pd.to_datetime(log_transformed_df.index)

# check if the index is a datetime
display(log_transformed_df.index)



# In[ ]:


display(log_transformed_df.head())


# In[ ]:


display(quarterly_data)
# The below columns all dislay for us quarterly data, we will need to deal with these NaN values in a different way than the rest of the Nan values, as these are 
# correct in every way, the only concern is that it could be made more robust... 


# # Fixing Quarterly data: Reverse Imputation
# 
# The dataset contains quarterly data, and the NaN values are filled with the previous quarter's data. This is a common practice in financial data, but it can lead to data leakage.

# In[ ]:


df = log_transformed_df.copy()


# In[ ]:


# Step 1: Identify Numeric Columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Calculate Summary Statistics
summary_statistics = df[numeric_columns].describe()

# Step 3: Look for Outliers
# Define a threshold for identifying overly large values, e.g., 3 standard deviations from the mean
threshold = 3 * summary_statistics.loc['std']

# Identify overly large values
overly_large_values = (df[numeric_columns] > (summary_statistics.loc['mean'] + threshold)).any()

# Print columns with overly large values
print("Columns with overly large values:")
print(overly_large_values[overly_large_values].index.tolist())


# ---

# # Removing NaN values from GDP

# # 0. Prepare Testing Functions

# In[ ]:


def evaluate_imputation(original_data, imputed_data):
    # Extracting values and aligning indexes
    original_values = original_data.values
    imputed_values = imputed_data.values
    index = original_data.index

    # Compute absolute error between original and imputed data
    abs_error = np.abs(original_values - imputed_values)

    # Compute mean absolute error
    mean_abs_error = abs_error.mean()

    # Plot original and imputed data
    plt.figure(figsize=(12, 6))
    plt.plot(index, original_values, label='Original Data', marker='o', linestyle='-')
    plt.plot(index, imputed_values, label='Imputed Data', marker='x', linestyle='-')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('GDP')
    plt.title('Comparison of Original and Imputed GDP Data')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print the number of NaN
    print("Number of NaN values in the original data:", original_data.isnull().sum())
    print("Number of NaN values in the imputed data:", imputed_data.isnull().sum()) 

    return mean_abs_error

def matplotlib_comparison_plot(original_data, imputed_data, column):
    # Plot original GDP
    original_plot = original_data[column].hvplot(line_color='blue', width=800, height=400, label=f'Original {column}')

    # Plot imputed GDP
    imputed_plot = imputed_data[column].hvplot(line_color='red', width=800, height=400, label=f'Imputed {column}')

    # Overlay the two plots
    overlay_plot = original_plot * imputed_plot

    # Customize the overlay plot with additional options
    final_plot = overlay_plot.opts(
        title=f"Original vs. Imputed {column} Data",
        xlabel='Date',
        ylabel=column,
        legend_position='top_left',
        tools=['pan', 'box_zoom', 'save']
    )

    # Display the interactive plot
    display(final_plot)



# # 1. Simple Imputation

# In[ ]:


from sklearn.impute import SimpleImputer

def impute_missing_values_mean(df, column):
    train_df = df.copy()

    # Data with known values for the target column
    reg_train_data = train_df.dropna(subset=[column])

    # Data with missing values for the target column
    predict_data = train_df[train_df[column].isnull()]

    # Features and target for training
    features = [col for col in train_df.columns if col != column]
    X_train = reg_train_data[features]
    y_train = reg_train_data[column]

    # Initialize the imputer
    imputer = SimpleImputer(strategy='mean')  # You can also try other strategies like 'median' or 'most_frequent'

    # Fit the imputer on the training data
    imputer.fit(X_train)

    # Impute missing values in the predict_data
    X_predict = predict_data[features]
    predicted_values = imputer.transform(X_predict)

    # Fill in the missing values in the original DataFrame
    predict_data[column] = predicted_values

    # Concatenate reg_train_data and predict_data
    imputed_data = pd.concat([reg_train_data, predict_data], axis=0)

    return imputed_data

# Example usage:
column = 'GDP'
gdp_imputed_mean = impute_missing_values_mean(log_transformed_df, column)
evaluate_imputation(log_transformed_df[column], gdp_imputed_mean[column])
# After imputation, ensure 'Date' is the index if it was ever reset
gdp_imputed_mean.index = pd.to_datetime(gdp_imputed_mean.index)  # Reinstate 'Date' as the index


# # Evaluate Simple Imputation

# In[ ]:


# Compare the original and imputed GDP data
matplotlib_comparison_plot(log_transformed_df, gdp_imputed_mean, 'GDP')


# # 2. Random Forest Imputation

# In[ ]:


def impute_missing_values_rf(df, column):
    train_df = df.copy()
    reg_train_data = train_df.dropna(subset=[column])
    predict_data = train_df[train_df[column].isnull()]
    features = [col for col in train_df.columns if col != column]
    X_train = reg_train_data[features]
    y_train = reg_train_data[column]
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    predict_data_imputed = imputer.transform(predict_data[features])
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_imputed, y_train)
    predicted_values = regressor.predict(predict_data_imputed)
    predict_data[column] = predicted_values
    imputed_data = pd.concat([reg_train_data, predict_data]).sort_index()
    return imputed_data

# Example usage:
column = 'GDP'
gdp_imputed_rf = impute_missing_values_rf(log_transformed_df, column)
evaluate_imputation(log_transformed_df[column], gdp_imputed_rf[column])
# After imputation, ensure 'Date' is the index if it was ever reset
gdp_imputed_rf.index = pd.to_datetime(gdp_imputed_rf.index)  # Reinstate 'Date' as the index


# In[ ]:


# Evaluate using matplotlib comparison
matplotlib_comparison_plot(log_transformed_df, gdp_imputed_rf, 'GDP')


# # 3. Spline Interpolation

# In[ ]:


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def impute_missing_values_spline(df, column):
    # Ensure the index is in datetime format and sort the data
    df = df.sort_index()
    
    # Extract the non-missing values to fit the spline
    known_data = df.dropna(subset=[column])
    known_index = known_data.index.map(pd.Timestamp.toordinal)  # Convert dates to ordinal
    
    # Fit a cubic spline using known data points
    cs = CubicSpline(known_index, known_data[column])
    
    # Apply the cubic spline to predict missing values
    missing_index = df[df[column].isnull()].index.map(pd.Timestamp.toordinal)
    predicted_values = cs(missing_index)
    
    # Fill in the missing values in the original DataFrame
    df.loc[df[column].isnull(), column] = predicted_values
    
    return df

# Example usage:
# Assuming 'log_transformed_df' is your DataFrame and it is indexed by a datetime index
log_transformed_df.index = pd.to_datetime(log_transformed_df.index)  # Convert index to datetime if not already
column = 'GDP'
gdp_imputed_spline = impute_missing_values_spline(log_transformed_df, column)
# Evaluate the imputation
evaluate_imputation(log_transformed_df[column], gdp_imputed_spline[column])


# In[ ]:


import matplotlib.pyplot as plt

# Original and imputed data
original_data = log_transformed_df['GDP']
imputed_data = gdp_imputed_spline['GDP']

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Plot the original data in the first subplot
ax1.plot(original_data.index, original_data.values, label='Original Data', marker='o', linestyle='-')
ax1.set_ylabel('GDP')
ax1.grid(True)
ax1.legend()

# Plot the comparison of original and imputed data in the second subplot
ax2.plot(original_data.index, original_data.values, label='Original Data', marker='o', linestyle='-', color='blue')
ax2.plot(imputed_data.index, imputed_data.values, label='Imputed Data', marker='x', linestyle='-', color='orange')
ax2.set_ylabel('GDP')
ax2.grid(True)
ax2.legend()

# Plot only the imputed data in the third subplot
ax3.plot(imputed_data.index, imputed_data.values, label='Imputed Data', marker='x', linestyle='-', color='orange')
ax3.set_ylabel('GDP')
ax3.grid(True)
ax3.legend()

# Rotate x-axis labels for better readability
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Set common title
plt.suptitle('Comparison of Original and Imputed GDP Data', y=0.95)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[ ]:


# Plot original GDP
original_plot = log_transformed_df['GDP'].hvplot(line_color='blue', width=800, height=400, label='Original GDP')

# Plot imputed GDP
imputed_plot = gdp_imputed_spline['GDP'].hvplot(line_color='red', width=800, height=400, label='Imputed GDP')

# Overlay the two plots
overlay_plot = original_plot * imputed_plot

# Customize the overlay plot with additional options
final_plot = overlay_plot.opts(
    title="Original vs. Imputed GDP Data",
    xlabel='Date',
    ylabel='GDP',
    legend_position='top_left',
    tools=['pan', 'box_zoom', 'save']
)

# Display the interactive plot
display(final_plot)


# In[ ]:


original_data = log_transformed_df
imputed_data = gdp_imputed_spline

# Plot original GDP
original_plot = original_data[column].hvplot(line_color='blue', width=800, height=400, label=f'Original {column}')

 # Plot imputed GDP
imputed_plot = imputed_data[column].hvplot(line_color='red', width=800, height=400, label=f'Imputed {column}')

    # Overlay the two plots
overlay_plot = original_plot * imputed_plot

    # Customize the overlay plot with additional options
final_plot = overlay_plot.opts(
    title=f"Original vs. Imputed {column} Data",
    xlabel='Date',
    ylabel=column,
    legend_position='top_left',
    tools=['pan', 'box_zoom', 'save']
)

# Display the interactive plot
display(final_plot)


# In[ ]:


# Update the original DataFrame using combine_first to fill in the NaN values


display(log_transformed_df.head())
display(gdp_imputed_spline.head())
# replace the original GDP column with the imputed values
log_transformed_df['GDP'] = log_transformed_df['GDP'].combine_first(gdp_imputed_spline['GDP'])
# log_transformed_df['GDP'] = combined_gdp
# Display
display(log_transformed_df.head())


# In[ ]:


# Ensure both DataFrames have the same date index and only compare where there are original values
common_index = original_data.dropna().index.intersection(imputed_data.dropna().index)

# Retrieve the 'GDP' series from both DataFrames based on the common index
original_gdp = original_data.loc[common_index, 'GDP']
imputed_gdp = imputed_data.loc[common_index, 'GDP']

# Calculate the correlation
correlation = original_gdp.corr(imputed_gdp)
print(f"The correlation coefficient between the original and imputed GDP data is: {correlation}")


# # **Evaluation of Imputation**:
# 
# ### Upon further evaluation, it is clear that, for quarterly data, the cubic spline technique is the best choice for imputation. It is the most accurate and the least biased. We see large deviations with the mean, and the random forest imputation that make the data largely useless. The accuracy with which the cubic spline imputation can predict the GDP is impressive, at first I couldn't spot it's graph, because it was that close to the real data values.
# 
# # *We will then proceed to impute the GDP data using the cubic spline technique.*
# 

# ---

# # Automating the Imputation for Multiple Columns:
# 
# ### We impute all the quarterly data columns using the cubic spline technique. We will then evaluate the imputation by comparing the imputed data with the original data.

# In[ ]:


quarterly_columns = ['GDP', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'GDPCTPI', 'BSHCFAW']
original_data = log_transformed_df

# DataFrame to store imputed values
imputed_dataframes = {}

for column in quarterly_columns:
    if column in log_transformed_df.columns:
        imputed_data = impute_missing_values_spline(log_transformed_df, column)
        # replace the original GDP column with the imputed values
        log_transformed_df[column] = log_transformed_df[column].combine_first(imputed_data[column])
        #imputed_dataframes[column] = imputed_data
        print(f"Imputed {column} - Correlation with original: {log_transformed_df[column].corr(imputed_data[column])}")
        # Plot original GDP
        original_plot = original_data[column].hvplot(line_color='blue', width=800, height=400, label=f'Original {column}')

        # Plot imputed GDP
        imputed_plot = imputed_data[column].hvplot(line_color='red', width=800, height=400, label=f'Imputed {column}')

        # Overlay the two plots
        overlay_plot = original_plot * imputed_plot

        # Customize the overlay plot with additional options
        final_plot = overlay_plot.opts(
            title=f"Original vs. Imputed {column} Data",
            xlabel='Date',
            ylabel=column,
            legend_position='top_left',
            tools=['pan', 'box_zoom', 'save']
        )

        # Display the interactive plot
        display(final_plot)
    else:
        print(f"Column {column} does not exist in DataFrame.")


# In[ ]:


# Check how many Nan values there are in the dataset
display(quarterly_data)
display(log_transformed_df[quarterly_data].isnull().sum())


# In[ ]:


# Find the remaining NaN values
Nan_columns = find_Nan_columns(log_transformed_df)

# Display the columns that have NaN values
display(Nan_columns)
# Display the amount of Nan values in the columns
display(log_transformed_df[Nan_columns].isnull().sum())


# In[ ]:


# display isnull
display(log_transformed_df.isnull().sum())


# In[ ]:


# Save the cleaned data, to the economic_data.csv file
log_transformed_df.to_csv('economic_data.csv')

# check nan values 
display(log_transformed_df.isnull().sum())


# # We have completed the Quarterly data imputation. We will now proceed to clean the rest of the data.
# 
# ---

# # CLEANING THE REST OF THE DATA

# In[ ]:


# visualize the data as a scatter plot
fivefive_columns = ['PCE', 'PCEPI', 'PCEPILFE', 'HOUST', 'DSPI']
for column in fivefive_columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(log_transformed_df[column])), log_transformed_df[column])
    plt.title(f'{column} Scatter Plot')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()


# In[ ]:


for column in fivefive_columns:
    print(f"Missing data for {column}:")
    missing_data = log_transformed_df[log_transformed_df[column].isnull()]
    print(missing_data[[column]])  # Display only the column of interest
    print("\n")  # Add a newline for better readability between outputs


# # Removing even more NaN values
#  
# 
# ### The rest of the NaN values don't seem to share any numeric pattern, except for the DGS5, and DGS10 columns, which both contain 304 NaN values. DGS2 has 413, while DTWEXBGS has a whopping 638, while ICS_ALL has 154. 

# In[ ]:


# plot all the columns in is.null
log_transformed_df['DGS10'].plot()
log_transformed_df['DGS2'].plot()
log_transformed_df['DGS5'].plot()
log_transformed_df['DTWEXBGS'].plot()

# spline interpolate only DGS2, DGS5, DGS10, drop DTWEXBGS
log_transformed_df = log_transformed_df.drop(columns=['DTWEXBGS'])


# In[ ]:


log_transformed_df['DGS10'].plot()
log_transformed_df['DGS2'].plot()
log_transformed_df['DGS5'].plot()


# In[ ]:


# Apply the spline interpolation to the 'DGS10', DGS2, AND DGS5 columns
imputed_logged_data = impute_missing_values_spline(log_transformed_df, 'DGS10')
imputed_logged_data = impute_missing_values_spline(log_transformed_df, 'DGS2')
imputed_logged_data = impute_missing_values_spline(log_transformed_df, 'DGS5')

# Check if the interpolation was successful (wasn't)
imputed_logged_data['DGS10'].plot()

# Find the ranges 'daterangeof the columns
columns  = ['DGS10', 'DGS2', 'DGS5']

display(log_transformed_df[columns].head())

# let us try and cut the data. 
# save the columns as a new dtaframe with the date
df = log_transformed_df[columns].copy()
# save as csv -- treasury_yield.csv
df.to_csv('treasury_yield.csv')


# # Plan Moving Forward:
# ## Step 1: Interpolate the Treasury Note Data from 1976-2024
# Since you have a relatively complete dataset from 1976 onwards, start by using spline interpolation to handle missing values within this range for the DGS2, DGS5, and DGS10 series. This will give you a solid foundation of complete data for the more recent periods.

# In[ ]:


# Find the number of missing values after 1976
missing_data = df[df.index.year >= 1976].isnull().sum()
display(missing_data)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Function to visualize NaN values in DataFrame
def plot_nan_values(dataframe, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataframe.isnull(), cbar=False, cmap='viridis')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()



# In[ ]:


original_data = pd.read_csv('economic_data.csv', index_col='Date')
# Convert index to DateTimeIndex
original_data.index = pd.to_datetime(original_data.index)

# Treasury yield columns to be imputed
treasury_yield_columns = ['DGS2', 'DGS5', 'DGS10']

# DataFrame to store imputed values
imputed_dataframes = {}

for column in treasury_yield_columns:
    if column in original_data.columns:
        # Slice the DataFrame to include data from 1976 to 2024
        sliced_data = original_data.loc['1976-01-01':'2024-01-01', :]

        # Perform spline interpolation on the sliced DataFrame
        imputed_data = impute_missing_values_spline(sliced_data, column)
        
        # Merge the imputed values back into the original DataFrame
        original_data[column] = original_data[column].combine_first(imputed_data[column])
        
        # Store the imputed DataFrame
        imputed_dataframes[column] = imputed_data
        
        print(f"Imputed {column} - Correlation with original: {original_data[column].corr(imputed_data[column])}")
        
        # Plot original and imputed data
        original_plot = original_data[column].hvplot(line_color='blue', width=800, height=400, label=f'Original {column}')
        imputed_plot = imputed_data[column].hvplot(line_color='red', width=800, height=400, label=f'Imputed {column}')
        overlay_plot = original_plot * imputed_plot
        final_plot = overlay_plot.opts(
            title=f"Original vs. Imputed {column} Data",
            xlabel='Date',
            ylabel=column,
            legend_position='top_left',
            tools=['pan', 'box_zoom', 'save']
        )
        display(final_plot)
    else:
        print(f"Column {column} does not exist in DataFrame.")

# Save the spliced data as a new CSV file
sliced_data.to_csv('spliced_data.csv')



# In[ ]:


# Add the imputed values to the dataframe, these are the new values for the column
original_data['DGS2'] = original_data['DGS2'].combine_first(imputed_data['DGS2'])
original_data['DGS5'] = original_data['DGS5'].combine_first(imputed_data['DGS5'])
original_data['DGS10'] = original_data['DGS10'].combine_first(imputed_data['DGS10'])

# Save the cleaned data to a new CSV file
original_data.to_csv('economic_data.csv')

# display the tail of the data
display(original_data[treasury_yield_columns].tail())
display(original_data[treasury_yield_columns].isnull().sum())


# In[ ]:


# Interpolate again, now for DGS10, and DGS5 only, from 1976-02-01:1976-01-01
# Perform spline interpolation on the sliced DataFrame

# Treasury yield columns to be imputed
treasury_yield_columns = ['DGS5', 'DGS10']

# DataFrame to store imputed values
imputed_dataframes = {}

for column in treasury_yield_columns:
    if column in original_data.columns:
        # Slice the DataFrame to include data from 1976 to 2024
        sliced_data = original_data.loc['1962-02-01':'1976-01-01', :]

        # Perform spline interpolation on the sliced DataFrame
        imputed_data = impute_missing_values_spline(sliced_data, column)
        
        # Merge the imputed values back into the original DataFrame
        original_data[column] = original_data[column].combine_first(imputed_data[column])
        
        # Store the imputed DataFrame
        imputed_dataframes[column] = imputed_data
        
        print(f"Imputed {column} - Correlation with original: {original_data[column].corr(imputed_data[column])}")
        
        # Plot original and imputed data
        original_plot = original_data[column].hvplot(line_color='blue', width=800, height=400, label=f'Original {column}')
        imputed_plot = imputed_data[column].hvplot(line_color='red', width=800, height=400, label=f'Imputed {column}')
        overlay_plot = original_plot * imputed_plot
        final_plot = overlay_plot.opts(
            title=f"Original vs. Imputed {column} Data",
            xlabel='Date',
            ylabel=column,
            legend_position='top_left',
            tools=['pan', 'box_zoom', 'save']
        )
        display(final_plot)
    else:
        print(f"Column {column} does not exist in DataFrame.")

# Save the spliced data as a new CSV file
sliced_data.to_csv('spliced_data.csv')


# In[ ]:


# Add the imputed values to the dataframe, these are the new values for the column
original_data['DGS2'] = original_data['DGS2'].combine_first(imputed_data['DGS2'])
original_data['DGS5'] = original_data['DGS5'].combine_first(imputed_data['DGS5'])
original_data['DGS10'] = original_data['DGS10'].combine_first(imputed_data['DGS10'])

# Save the cleaned data to a new CSV file
original_data.to_csv('economic_data.csv')


# In[ ]:


df_dummy = pd.read_csv('economic_data.csv')

# Convert the index to DateTimeIndex
df_dummy.index = pd.to_datetime(df_dummy.index)
# check for NaN values before 1976
missing_data = df_dummy[df_dummy.index.year < 1976].isnull().sum()
print("1.Before 1976")
display(missing_data)

# check for NaN values after 1976
print("2. After 1976")
missing_data = df_dummy[df_dummy.index.year >= 1976].isnull().sum()
display(missing_data)


# # Step 2. Model Development and Validation
# 
# ## Once you have interpolated the data from 1976-2024:
# 
# 1. Develop the VAR Model: Use this dataset to develop and validate your VAR model. Ensure that the model fits well and that the predictions are reasonable by comparing them against known data points and checking for errors such as residuals.
# 2. Model Validation: This step is critical before proceeding with forecasting historical missing data to ensure the model’s stability and accuracy.
# 

# # Dropping DTWEXBGS: The Nominal Broad U.S. Dollar Index,
# 
# ### We are only dropping to simplify the calculation and also because I don't know of any reliable imputation technique that could be used to impute the data. The data is missing points from 1960-2006, and it is not worth the effort to impute the data. Even if we did, the data would likely be full ofbias and innacuracies. 
# ### I would have considered changing, only if we were to use the data from 2006-2024. In a segment analysis of the data, we could have used the data from 2006-2024, which may be a next step, before which we will impute the data.
# ### To prepare for potentially using the data from 2006-2024, we will impute the data using the cubic spline technique.

# ## Removing NaN values: 
# 
# ### It is critical that we remove NaN vaues, -- we will be holistically evaluating the data. We will start with ensuring that NaN values are not simply removed, but are replaced with either the mean of the column, or with new values based on predictive imputation. *We will be using the latter, as it is more accurate.*

# # Creating final data:
# 
# #### We will be creating our final dataset, that we will be using for our initial testing and preprocessing.

# In[ ]:


# Save the 

