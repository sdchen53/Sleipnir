# transfer the data shape from stlouisfed to alphavantage format
import pandas as pd
import os



def transform_stlouisfed_to_alphavantage(input_file, name, output_file):
    """
    stlouisfed: observation_date, NAME
    alphavantage: timestamp,open,high,low,close,volume
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    # Read the input CSV file
    df = pd.read_csv(input_file, parse_dates=['observation_date'])
    
    # Rename columns to match alphavantage format
    df.rename(columns={'observation_date': 'timestamp', name: 'close'}, inplace=True)
    
    # Add dummy columns for open, high, low, and volume
    df['open'] = df['close']
    df['high'] = df['close']
    df['low'] = df['close']
    df['volume'] = 0  # Assuming volume is not available
    
    # Reorder columns to match alphavantage format
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save the transformed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    name = 'DJIA'  # Replace with the actual column name you want to transform
    input_file = f'data/{name}.csv'  # Replace with your actual input file path
    output_file = f'data/stock_market_data-{name}.csv'  # Desired output file path
    
    transform_stlouisfed_to_alphavantage(input_file, name, output_file)