import os

import pandas as pd
import urllib3


def getStockMarketData(ticker, api_key = 'F62MHL3VDUMCEMGP'):
    if not os.path.exists(f'./data/stock_market_data-{ticker}.csv'):
        # ====================== Loading Data from Alpha Vantage ==================================


        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s&datatype=csv"%(ticker,api_key)

        # Save data to this file
        file_to_save = './data/stock_market_data-%s.csv'%ticker

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            http = urllib3.PoolManager()
            with http.request('GET', url_string)  as url:
                print("Response from Alpha Vantage:", url.getcode())
                data = pd.read_csv(url)
                print("Data fetched from Alpha Vantage:", data.head())
                # Save the DataFrame to a CSV file
                data.to_csv(file_to_save, index=False)
                print('Data saved to : %s' % file_to_save)

    else:
        # If you have already saved data, just load it from the file
        df = pd.read_csv(f'./data/stock_market_data-{ticker}.csv', parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df
