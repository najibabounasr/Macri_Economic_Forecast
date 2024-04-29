import pandas as pd

def convert_csv(input_filepath, output_filepath):
    # Load the data
    df = pd.read_csv(input_filepath)
    
    # Create a new 'Date' column by combining 'Year' and 'Period'
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Period'].str[1:], format='%Y%m')
    
    # Select only the 'Date' and 'Value' columns
    result_df = df[['Date', 'Value']]
    
    # Write the result to a new CSV file
    result_df.to_csv(output_filepath, index=False)
    
    print(f"Converted CSV saved to {output_filepath}")


input4 = input("Enter the input file path: ")
output = input("Enter the output file path: ")

convert_csv(input4, output)
