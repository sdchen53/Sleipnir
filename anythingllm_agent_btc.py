import json
import os
import re
import sys
import urllib3
import argparse
import pandas as pd
import random
import datetime as dt
from tqdm import tqdm

from myLogger import myLogger

# try auth
def auth() -> str:
    http = urllib3.PoolManager()
    url = f"{entrypoint}auth"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    response = http.request("GET", url, headers=headers)
    if response.status == 200:
        return response.data.decode("utf-8")
    else:
        raise Exception(
            f"Authentication failed: {response.status} {response.data.decode('utf-8')}")


def generateNewWorkspace(name: str) -> str:
    http = urllib3.PoolManager()
    url = f"{entrypoint}workspace/new"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    body = {
        "name": name,
        "similarityThreshold": 0.7,
        "openAiTemp": 0.7,
        "openAiHistory": 20,
        "openAiPrompt": "Custom prompt for responses",
        "queryRefusalResponse": "Custom refusal message",
        "chatMode": "chat",
        "topN": 4
    }
    response = http.request("POST", url, headers=headers,
                            body=json.dumps(body).encode('utf-8'))
    # response should be like:
    # '''
    #     {
    #     "workspace": {
    #         "id": 79,
    #         "name": "Sample workspace",
    #         "slug": "sample-workspace",
    #         "createdAt": "2023-08-17 00:45:03",
    #         "openAiTemp": null,
    #         "lastUpdatedAt": "2023-08-17 00:45:03",
    #         "openAiHistory": 20,
    #         "openAiPrompt": null
    #     },
    #     "message": "Workspace created"
    #     }
    # '''
    if response.status == 200:
        try:
            # get the workspace ID from the response
            workspace_data = response.data.decode("utf-8")
            workspace_json = json.loads(workspace_data)
            workspace_slug = workspace_json['workspace']['slug']
            return workspace_slug
        except json.JSONDecodeError:
            raise Exception(
                f"Failed to parse workspace creation response: {response.data.decode('utf-8')}")
    else:
        raise Exception(
            f"Workspace creation failed: {response.status} {response.data.decode('utf-8')}")


def deleteWorkspace(slug: str) -> str:
    http = urllib3.PoolManager()
    url = f"{entrypoint}workspace/{slug}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    response = http.request("DELETE", url, headers=headers)
    if response.status == 200:
        return response.data.decode("utf-8")
    else:
        raise Exception(
            f"Workspace deletion failed: {response.status} {response.data.decode('utf-8')}")


def generateNewThread(workspace_slug: str) -> str:
    http = urllib3.PoolManager()
    url = f"{entrypoint}workspace/{workspace_slug}/thread/new"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    response = http.request("POST", url, headers=headers)
    if response.status == 200:
        try:
            # get the thread ID from the response
            thread_data = response.data.decode("utf-8")
            thread_json = json.loads(thread_data)
            thread_slug = thread_json['thread']['slug']
            return thread_slug
        except json.JSONDecodeError:
            raise Exception(
                f"Failed to parse thread creation response: {response.data.decode('utf-8')}")


def deleteThread(workspace_slug: str, thread_slug: str) -> bool:
    http = urllib3.PoolManager()
    url = f"{entrypoint}workspace/{workspace_slug}/thread/{thread_slug}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    response = http.request("DELETE", url, headers=headers)
    if response.status == 200:
        return True
    else:
        raise Exception(
            f"Thread deletion failed: {response.status} {response.data.decode('utf-8')}")


def chatWithThread(workspace_slug: str, thread_slug: str, message: str, timeout=60) -> str:
    http = urllib3.PoolManager()
    url = f"{entrypoint}workspace/{workspace_slug}/thread/{thread_slug}/chat"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {anything_api}"
    }
    body = {
        "message": message,
        "mode": "chat",
        "userId": 1,
        # "attachments": [
        #     {
        #     "name": "image.png",
        #     "mime": "image/png",
        #     "contentString": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        #     }
        # ],
        "reset": False
    }
    response = http.request("POST", url, headers=headers,
                            body=json.dumps(body).encode('utf-8'), timeout=timeout)
    # chat should be like:
    # '''
    # {
    #     "id": "a0982a32-3a52-49f3-a9a6-7a6d165e9c1a",
    #     "type": "textResponse",
    #     "close": true,
    #     "error": null,
    #     "chatId": 51,
    #     "textResponse": "Hello! I received your test message loud and clear. How can I assist you today? 😊  \n\n(Note: This is a standard acknowledgment. If you need help with something specific, feel free to ask!)",
    #     "sources": [],
    #     "metrics": {
    #         "prompt_tokens": 23,
    #         "completion_tokens": 42,
    #         "total_tokens": 65,
    #         "outputTps": 1.2363486503193901,
    #         "duration": 33.971
    #     }
    # }
    # '''

    if response.status == 200:
        try:
            chat_data = response.data.decode("utf-8")
            chat_json = json.loads(chat_data)
            if 'textResponse' in chat_json:
                return chat_json['textResponse']
            else:
                raise Exception(
                    f"Chat response does not contain 'textResponse': {chat_data}")
        except json.JSONDecodeError:
            raise Exception(
                f"Failed to parse chat response: {response.data.decode('utf-8')}")
    else:
        raise Exception(
            f"Chat with thread failed: {response.status} {response.data.decode('utf-8')}")
    
def extract_prediction_line(response_text):
    # Pattern to match the prediction line 
    # AAPL, 2024-05-01, buy-in, 0.2, hold, 0.6, sell-out, 0.2, hands-in, 0, hands-off, 0
    # `AAPL, 2024-05-01, buy-in, 0.3, hold, 0.6, sell-out, 0.1, hands-in, 150, hands-off, 350`
    pattern = r'([A-Z]+),\s*(\d{4}-\d{2}-\d{2}),\s*buy-in,\s*([\d.]+),\s*hold,\s*([\d.]+),\s*sell-out,\s*([\d.]+),\s*hands-in,\s*(\d+),\s*hands-off,\s*(\d+)'
    
    match = re.search(pattern, response_text)
    if match:
        return match.group(0).strip().replace('*', '').replace('`', '')  # Returns the full matched line
    
    return None

def extract_confidence_operation(prediction_line):
    """
    Extract the operation with highest confidence from prediction line
    """
    # Parse the prediction line
    # "Current holding is {} shares, max holding is {} shares.  Answer Example: '{}, buy-in, 0.5, hold, 0.1, sell-out, 0.4, hands-in, 200, hands-off, 100'"
    # AAPL, 2024-05-01, buy-in, 0.1, hold, 0.6, sell-out, 0.3, hands-in, 0, hands-off, 500
    print(f"Extracting from prediction line: {prediction_line}")
    pattern = r'([A-Z]+),\s*(\d{4}-\d{2}-\d{2}),\s*buy-in,\s*([\d.]+),\s*hold,\s*([\d.]+),\s*sell-out,\s*([\d.]+),\s*hands-in,\s*(\d+),\s*hands-off,\s*(\d+)'
    
    match = re.search(pattern, prediction_line)
    if not match:
        raise ValueError("Invalid prediction line format")
    
    ticker, date, buy_confidence, hold_confidence, sell_confidence, hands_in, hands_off = match.groups()
    
    # Convert to float for comparison
    buy_conf = float(buy_confidence)
    hold_conf = float(hold_confidence)
    sell_conf = float(sell_confidence)
    # hands_in = int(hands_in.split(',')[1].strip())
    # hands_off = int(hands_off.split(',')[1].strip())
    hands_in = int(hands_in)
    hands_off = int(hands_off)
    
    # Find the operation with highest confidence
    confidences = {
        'buy': buy_conf,
        'hold': hold_conf,
        'sell': sell_conf
    }
    # return confidences

    highest_operation = max(confidences, key=lambda k: confidences[k])
    
    # get corresponding hands-in or hands-off
    if highest_operation == 'buy':
        hands = hands_in
    elif highest_operation == 'sell':
        hands = hands_off
    else:
        hands = 0  # Default to 0 for hold or if no hands specified
    
    # Convert to numeric representation
    operation_map = {
        'buy': 1,
        'hold': 0,
        'sell': -1
    }
    
    return operation_map[highest_operation], hands

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





quantitive_factor = """
You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.

## Investment Focus
- Small-cap stocks (market cap < 1B USD)
- Value factors: low P/E, P/B, P/S ratios
- Quality metrics: ROE, ROA, debt levels

## Decision Framework
**BUY Signals:**
- Strong value metrics vs peers/history
- Solid fundamentals despite small size
- Positive alpha potential

**SELL Signals:**
- Valuation reaches fair value
- Deteriorating fundamentals
- Better opportunities available

## Response Format
**Stock**: [Ticker/Name]
**Price**: [Current] | **Market Cap**: [Value]

**Scores** (1-10):
- Size Factor: [Score]
- Value Factor: [Score] 
- Quality Factor: [Score]

**Decision**: [BUY/HOLD/SELL]
**Confidence**: [1-10]
**Target Allocation**: [%]
**Expected Return**: [% over timeframe]

**Rationale**: [1-2 key points]
**Key Risk**: [Main concern]
**Entry/Exit**: [Price levels]

Note: Size factor effectiveness in A-shares has weakened post-2017.
"""

bollinger_bands = """
You are a Bollinger Bands mean reversion trader. Your strategy: buy when price breaks below lower band, sell when price breaks above upper band, exit at middle band.

## Setup
- Period: 20 days
- Standard deviation: 2.0
- Upper Band = MA + (2 x StdDev)
- Lower Band = MA - (2 x StdDev)

## Signals
**BUY**: Price closes below lower band
**SELL**: Price closes above upper band  
**EXIT**: Price returns to middle band or 3% stop-loss

## Decision Format
**Symbol**: [Ticker]
**Signal**: [BUY/SELL/EXIT/HOLD]
**Current Price**: [Price]
**Band Position**: [Above/Below/Within bands]
**Entry/Exit Price**: [Specific level]
**Stop Loss**: [3% from entry]
**Rationale**: [1-2 sentences]

## Risk Rules
- Max 5% portfolio per trade
- No trading in strong trends
- Exit if no reversion within 10 days

Keep responses brief and actionable.
"""

camp_model = """
You are a CAPM model analyst. Your role is to determine when CAPM is appropriate for asset pricing and when multi-factor models are needed.

## CAPM Framework
**Formula**: E(Ri) = Rf + βi(E(Rm) - Rf)
- Ri = Expected return of asset i
- Rf = Risk-free rate
- βi = Beta (systematic risk)
- Rm = Market return

## Decision Criteria

**Use CAPM When:**
- Analyzing broad market portfolios
- Beta explains >70% of return variance
- No significant market anomalies present
- Short-term analysis (< 1 year)

**Use Multi-Factor Models When:**
- Significant alpha detected (α ≠ 0)
- Market anomalies present (size, value effects)
- Individual stock analysis
- Long-term analysis (> 1 year)

## Response Format
**Asset**: [Name/Ticker]
**Beta**: [Value]
**R-squared**: [% variance explained by market]

**Model Recommendation**: [CAPM/Multi-Factor]
**Reasoning**: [1-2 sentences why]

**If Multi-Factor Needed:**
- Additional factors: [Size/Value/Momentum/etc.]
- Expected alpha: [%]

**CAPM Limitations**: [Key assumptions violated]
"""

character_presets = {
    "base": "You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.",
    "quantitive_factor": quantitive_factor,
    "bollinger_bands": bollinger_bands,
    "camp_model": camp_model,
    # "final": "You are a final decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
    }
leader_preset = "You are a leader decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
anything_api = "1DV9A3A-SFFM1XR-QF4TYMR-HZ5X8RY"
entrypoint = "http://10.201.35.124:3001/api/v1/"
message_preset = "@agent Get {} stock info, from {} to {}, predict the later day's price, and give the buy-in or hold or sell-out decision on {}, with confidence."
message_outformat = "Current holding is {} shares, max holding is {} shares. Trade limit per operation is{}, expected return percentage is {}. Answer MUST contain Example style: '{}, buy-in, 0.5, hold, 0.1, sell-out, 0.4, hands-in, 200, hands-off, 100'"
# TODO 2025-07-20 git.V.aab24: fill the holding shares with updated operation
# TODO 2025-07-20 git.V.1029f: multi-agent, decision aggregation by leader, each agent has its own workspace and thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnythingLLM Agent CLI")
    parser.add_argument("--auth", action="store_true", help="Authenticate with AnythingLLM")
    parser.add_argument("--create-workspace", type=str, help="Create a new workspace with the given name")
    parser.add_argument("--create-thread", type=str, help="Create a new thread in the specified workspace")
    parser.add_argument("--api-key", type=str, help="API key for authentication", default=anything_api)
    parser.add_argument("--entrypoint", type=str, help="API entrypoint URL", default=entrypoint)
    # parser.add_argument("--message", type=str, help="Send a message to the specified thread", default="Hello, this is a test message.")
    parser.add_argument("--ticker", type=str, help="Ticker symbol for stock information", default="GOOG")
    parser.add_argument("--start-date", type=str, help="Start date for stock data in YYYY-MM-DD format", default="2024-05-01")
    parser.add_argument("--end-date", type=str, help="End date for stock data in YYYY-MM-DD format", default="2024-09-01")
    parser.add_argument("--date_steps", type=int, help="Number of steps to take in the simulation", default=3)
    parser.add_argument("--lookback-days", type=int, help="Number of days to look back for stock data", default=90)
    parser.add_argument("--delete-thread", type=str, help="Delete a thread in the specified workspace")
    parser.add_argument("--delete-workspace", type=str, help="Delete a workspace by slug")
    parser.add_argument("--init_holding", type=int, help="Initial holding shares", default=500)
    parser.add_argument("--max_holding", type=int, help="Maximum holding shares", default=1000)
    parser.add_argument("--trade_limit", type=int, help="Trade limit per operation", default=300)
    parser.add_argument("--return_expectation", type=float, help="Expected return percentage", default=0.1)
    args = parser.parse_args()

    anything_api = args.api_key
    entrypoint = args.entrypoint

    # daterange = pd.date_range(start=args.start_date, end=args.end_date, freq='B') # Business days only
    # load date range from stock market data
    df_stock = getStockMarketData(args.ticker)
    # Create a stepped date range
    daterange = pd.date_range(
        start=args.start_date, 
        end=args.end_date, 
        freq=f'{args.date_steps}D'  # Every x days
    )

    # Filter the dataframe to only include dates in the stepped range
    df_filtered = df_stock[df_stock.index.isin(daterange)]
    daterange = df_filtered.index
    # sort
    daterange = sorted(daterange)


    date_decision = {}
    current_holding = args.init_holding
    max_holding = args.max_holding

    # global anything_api, entrypoint  # IGNORE
    try:
        auth_response = auth()
        print(f"Authentication successful: {auth_response}")
    except Exception as e:
        print(f"Error during authentication: {e}")
        sys.exit(1)

    try:
        workspace_name = "My New Workspace"
        workspace_slug = generateNewWorkspace(workspace_name)
        print(f"Workspace created successfully: {workspace_slug}")
    except Exception as e:
        print(f"Error during workspace creation: {e}")
        sys.exit(1)

    logger = myLogger(name=str(os.getpid()), log_filename=workspace_slug, propagate=False)

    #%% create threads for each character
    character_slugs = {}
    for character, preset in character_presets.items():
        try:
            logger(f"Creating thread for character: {character}")
            thread_slug = generateNewThread(workspace_slug)
            logger(f"Thread created successfully: {thread_slug}")
            character_slugs[character] = thread_slug
        except Exception as e:
            logger.error_(f"Error during thread creation: {e}")

        # opening
        try:
            message = f"@agent {preset}"
            chat_response = chatWithThread(workspace_slug, character_slugs[character], message)
            logger(f"Chat response: {chat_response}")
        except Exception as e:
            logger.error_(f"Error during chat: {e}")

    #%% create leader thread
    try:
        leader_thread_slug = generateNewThread(workspace_slug)
        logger(f"Leader thread created successfully: {leader_thread_slug}")
        # opening
        message = f"@agent {leader_preset}"
        chat_response = chatWithThread(workspace_slug, leader_thread_slug, message)
        logger(f"Chat response: {chat_response}")
    except Exception as e:
        logger.error_(f"Error during final thread creation: {e}")

    
    #%% complete
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = f"{results_dir}/{workspace_slug}_{args.ticker}.csv"
    date_decision = {}

    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(results_file):
        pd.DataFrame(columns=['timestamp', 'Decision', 'Hands']).to_csv(results_file, index=False)

    pbar = tqdm(daterange, desc="Processing dates", unit="date")
    for date in pbar:
        date_str = date.strftime('%Y-%m-%d')
        logger(f"Processing date: {date_str}")
        
        # # Check if this date was already processed (resume capability)
        # existing_results = pd.read_csv(results_file)
        # if date_str in existing_results['timestamp'].values:
        #     logger(f"Date {date_str} already processed, skipping...")
        #     # Load existing decision for current_holding calculation
        #     existing_row = existing_results[existing_results['timestamp'] == date_str].iloc[0]
        #     current_holding = max(current_holding + existing_row['Hands'] * existing_row['Decision'], 0)
        #     continue
        
        lookback_start_date = (date - pd.Timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')
        logger(f"Lookback start date: {lookback_start_date}")
        
        message = message_preset.format(
            args.ticker, lookback_start_date, date_str, date_str, 
        ) + message_outformat.format(current_holding, max_holding,
            args.trade_limit, args.return_expectation,
            f"{args.ticker}, {date_str}"
        )
        
        try:
            response_list = []
            for character, thread_slug in character_slugs.items():
                logger(f"Sending message to {character}: {message}")
                try:
                    chat_response = chatWithThread(workspace_slug, thread_slug, message)
                    logger(f"Chat response from {character}: {chat_response}")
                    response_list.append(chat_response)
                except Exception as e:
                    logger.error_(f"Error during chat with {character}: {e}")
                    response_list.append(f"Error from {character}: {str(e)}")
            
            # Aggregate responses
            logger("Aggregating responses from all characters...")
            aggregated_response = "\n".join(response_list)
            
            try:
                logger("Sending aggregated response to final thread")
                chat_response = chatWithThread(workspace_slug, leader_thread_slug, 
                                            aggregated_response + message_outformat.format(
                                                current_holding, max_holding, 
                                                args.trade_limit, args.return_expectation,
                                                f"{args.ticker}, {date_str}"
                                            ))
                logger(f"Chat response: {chat_response}")
                
                one_line = extract_prediction_line(chat_response)
                decision, hands = extract_confidence_operation(one_line)
                logger(f"Extracted decision on date {date_str}: {decision} from response: {one_line}")
                
                current_holding = max(current_holding + hands * decision, 0)
                
                # Save result immediately
                new_result = pd.DataFrame({
                    'timestamp': [date_str],
                    'Decision': [decision],
                    'Hands': [hands]
                })
                
                # Append to existing results
                existing_results = pd.read_csv(results_file)
                updated_results = pd.concat([existing_results, new_result], ignore_index=True)
                updated_results.to_csv(results_file, index=False)
                
                logger(f"Results saved for date {date_str}: Decision={decision}, Hands={hands}")
                
            except Exception as e:
                logger.error_(f"Error during aggregation chat for date {date_str}: {e}")
                # Save error result
                error_result = pd.DataFrame({
                    'timestamp': [date_str],
                    'Decision': [0],  # Default to no action on error
                    'Hands': [0]
                })
                existing_results = pd.read_csv(results_file)
                updated_results = pd.concat([existing_results, error_result], ignore_index=True)
                updated_results.to_csv(results_file, index=False)
                
        except Exception as e:
            logger.error_(f"Critical error processing date {date_str}: {e}")
            # Save error result
            error_result = pd.DataFrame({
                'timestamp': [date_str],
                'Decision': [0],
                'Hands': [0]
            })
            existing_results = pd.read_csv(results_file)
            updated_results = pd.concat([existing_results, error_result], ignore_index=True)
            updated_results.to_csv(results_file, index=False)

    # Cleanup (moved outside the loop)
    try:
        delete_response = deleteWorkspace(workspace_slug)
        logger(f"Workspace deleted successfully: {delete_response}")
    except Exception as e:
        logger.error_(f"Error during workspace deletion: {e}")

    # Final results are already saved incrementally
    final_results = pd.read_csv(results_file)
    print(f"Final decisions saved to {results_file}")
    print(f"Total processed dates: {len(final_results)}")