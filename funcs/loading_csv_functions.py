import os   
import pandas as pd


def load_and_process_cpi_data(base_path):
    # List of CPI metrics
    cpi_metrics = ['CPI-Energy', 'CPI-Housing', 'CPI-Medical', 'CPI-Transportation', 'CPI-Universal', 'CPI-Core']
    
    # Initialize an empty DataFrame to store merged results
    merged_df = None
    
    for metric in cpi_metrics:
        # Construct the file path based on the described naming conventions
        directory_name = metric
        file_name = f'{metric.lower().replace("-", "_")}.csv'
        file_path = os.path.join(base_path, directory_name, file_name)
        
        # Check if the file exists before proceeding
        if os.path.exists(file_path):
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Clean the 'Value' column, removing any non-numeric characters from the values
            df['Value'] = pd.to_numeric(df['Value'].astype(str).str.replace(r'[^0-9\.]', '', regex=True), errors='coerce')
            
            # Rename the 'Value' column to include the CPI metric type for clarity when merged
            df.rename(columns={'Value': metric}, inplace=True)
            
            # If this is the first DataFrame, initialize merged_df
            if merged_df is None:
                merged_df = df
            else:
                # Merge the current DataFrame with the merged DataFrame on the 'Date' column
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        else:
            print(f"Warning: No file found at {file_path}")
    
    return merged_df


def merge_new_data(cpi_data, new_data, new_name, new_data_col_name):
    # Convert 'DATE' columns to datetime
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
    new_data['DATE'] = pd.to_datetime(new_data['DATE'])
    
    # Merge the dataframes on the 'Date' column
    merged_data = pd.merge(cpi_data, new_data, left_on='Date', right_on='DATE', how='left')
    
    # Drop the extra 'DATE' column from the new data
    merged_data.drop('DATE', axis=1, inplace=True)

    # Rename the new data column to 'Real GDP'
    merged_data.rename(columns={new_data_col_name: new_name}, inplace=True)
    return merged_data

# AND: (FOR LATER)

def merge_new_data_and_apply_pct_change(cpi_data, new_data, new_name, new_data_col_name):
    # Convert 'DATE' columns to datetime
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
    new_data['DATE'] = pd.to_datetime(new_data['DATE'])
    
    # Apply percentage change to the specific column in new_data
    new_data[new_data_col_name] = new_data[new_data_col_name].pct_change()
    
    # Merge the dataframes on the 'Date' column
    merged_data = pd.merge(cpi_data, new_data, left_on='Date', right_on='DATE', how='left')
    
    # Drop the extra 'DATE' column from the new data
    merged_data.drop('DATE', axis=1, inplace=True)

    # Rename the new data column to the specified new name
    merged_data.rename(columns={new_data_col_name: new_name}, inplace=True)
    
    return merged_data


def prepare_cpi_data(df):
    # List of CPI columns to convert and compute percent change
    cpi_columns = [
        'CPI-Energy', 'CPI-Housing', 'CPI-Medical', 
        'CPI-Transportation', 'CPI-Universal', 'CPI-Core'
    ]
    
    # Convert all specified CPI columns to numeric, coercing errors
    for column in cpi_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Calculate percent change for each CPI column and replace the original columns
    #for column in cpi_columns:
    #   df[column] = df[column].pct_change()

    return df

def preprocess_and_merge(data, other_data, column_on_which_to_merge):
    # Convert 'Month' and 'YYYY' into a 'Date' column in datetime format
    data['Date'] = pd.to_datetime(data['Month'] + ' ' + data['YYYY'].astype(str))
    
    # Drop the original 'Month' and 'YYYY' columns as they are no longer needed
    data.drop(['Month', 'YYYY'], axis=1, inplace=True)
    
    # Merge with the other dataframe on the specified column
    merged_data = pd.merge(data, other_data, left_on='Date', right_on=column_on_which_to_merge, how='left')
    
    return merged_data