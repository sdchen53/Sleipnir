import requests

url = "https://www.alphavantage.co/query"
params = {
    "function": "NEWS_SENTIMENT",
    "topics": "technology",
    "sort": "LATEST",
    "time_from": "20220410T0130",
    "time_to": "20220510T0230",
    "tickers": "AAPL,MSFT",
    "apikey": "F62MHL3VDUMCEMGP",
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()
print(data)