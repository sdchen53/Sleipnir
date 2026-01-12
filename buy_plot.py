import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import urllib
import datetime as dt

def buy_plot(df_market, df_operation):
    date_range = pd.date_range(start=df_operation.index.min(), end=df_operation.index.max())
    df_market = df_market[df_market.index.isin(date_range)]
    plt.figure(figsize=(12, 6))
    plt.plot(df_market.index, df_market['Close'], label='Market Close Price', color='blue')
    plt.scatter(df_operation.index, df_operation['Buy'], label='Buy Operations', color='green', marker='^')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Buy Operations on Market Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def getStockMarketData(ticker, api_key = '4MFYSPOPV210V5WT'):
    if not os.path.exists(f'./data/stock_market_data-{ticker}.csv'):
        # ====================== Loading Data from Alpha Vantage ==================================


        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

        # Save data to this file
        file_to_save = './data/stock_market_data-%s.csv'%ticker

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)

    else:
        # If you have already saved data, just load it from the file
        df = pd.read_csv(f'./data/stock_market_data-{ticker}.csv', parse_dates=['Date'])
        df.set_index('Date', inplace=True)
    return df

def getOperation(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Operation file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

ticker = 'AAL'  # Example ticker

if __name__ == "__main__":
    # Get stock market data
    df_market = getStockMarketData(ticker)
    
    # Get operation data
    df_operation = getOperation(f'results/my-new-workspace-55798868_decisions.csv')
    print(df_operation)

    # Generate buy plot