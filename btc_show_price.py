import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getStockMarketData(ticker):
    '''
    timestamp,open,high,low,close,basevolume,usdtvolume
    '''
    # If you have already saved data, just load it from the file
    df = pd.read_csv(f'./data/stock_market_data-{ticker}.csv')
    
    # Convert Unix timestamp to datetime
    # If timestamp is in seconds (like 1751472000), convert directly
    # If timestamp is in milliseconds, divide by 1000 first
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Clean the data - remove rows where OHLC values are missing
    # Keep only rows where open, high, low, close are not null and not empty strings
    df = df.replace('', np.nan)  # Replace empty strings with NaN
    df = df.dropna(subset=['open', 'high', 'low', 'close'])  # Drop rows with missing OHLC data
    
    # Convert OHLC columns to numeric if they're not already
    numeric_columns = ['open', 'high', 'low', 'close', 'basevolume', 'usdtvolume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Optionally, you can also filter out rows where volume is 0 if that indicates invalid data
    # df = df[df['basevolume'] > 0]
    
    return df

def getOperation(file_path):
    '''
    timestamp,Decision,Hands
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Operation file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Clean operation data if needed
    df = df.replace('', np.nan)
    df = df.dropna()  # Remove rows with any missing values
    
    return df

df = getStockMarketData('BTC')
print(df.describe())

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='BTC Close Price', color='blue')
plt.title('BTC Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid()
plt.show()