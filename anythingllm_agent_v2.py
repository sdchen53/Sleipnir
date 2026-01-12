import json
import os
import re
import urllib3
import argparse
import pandas as pd
import random

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


def chatWithThread(workspace_slug: str, thread_slug: str, message: str) -> str:
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
                            body=json.dumps(body).encode('utf-8'))
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
    
def extract_prediction_line(response_text, ticker=None, date=None):
    # Pattern to match the prediction line
    pattern = r'([A-Z]+),\s*(\d{4}-\d{2}-\d{2}),\s*buy-in,\s*([\d.]+),\s*hold,\s*([\d.]+),\s*sell-out,\s*([\d.]+)'
    
    match = re.search(pattern, response_text)
    if match:
        return match.group(0).strip().replace('*', '').replace('`', '')  # Returns the full matched line
    
    return None



anything_api = "1DV9A3A-SFFM1XR-QF4TYMR-HZ5X8RY"
entrypoint = "http://10.201.35.124:3001/api/v1/"
message_preset = "@agent {} Get {} stock info, from {} to {}, predict the later day's price, and give the buy-in or hold or sell-out decision on {}, with confidence. Current holding is {} shares.  Example: '{}, buy-in, 0.8, hold, 0.1, sell-out, 0.1'"
# TODO 2025-07-20 git.V.aab24: fill the holding shares with updated operation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnythingLLM Agent CLI")
    parser.add_argument("--auth", action="store_true", help="Authenticate with AnythingLLM")
    parser.add_argument("--create-workspace", type=str, help="Create a new workspace with the given name")
    parser.add_argument("--create-thread", type=str, help="Create a new thread in the specified workspace")
    parser.add_argument("--api-key", type=str, help="API key for authentication", default=anything_api)
    parser.add_argument("--entrypoint", type=str, help="API entrypoint URL", default=entrypoint)
    # parser.add_argument("--message", type=str, help="Send a message to the specified thread", default="Hello, this is a test message.")
    parser.add_argument("--character_setting", type=str, help="Character setting for the agent", default="You are a stock market expert, providing insights and predictions based on historical data.")
    parser.add_argument("--ticker", type=str, help="Ticker symbol for stock information", default="GOOG")
    parser.add_argument("--start-date", type=str, help="Start date for stock data in YYYY-MM-DD format", default="2024-01-01")
    parser.add_argument("--end-date", type=str, help="End date for stock data in YYYY-MM-DD format", default="2024-02-01")
    parser.add_argument("--lookback-days", type=int, help="Number of days to look back for stock data", default=30)
    parser.add_argument("--delete-thread", type=str, help="Delete a thread in the specified workspace")
    parser.add_argument("--delete-workspace", type=str, help="Delete a workspace by slug")
    args = parser.parse_args()

    args.api_key = args.api_key or anything_api
    args.entrypoint = args.entrypoint or entrypoint

    daterange = pd.date_range(start=args.start_date, end=args.end_date, freq='B') # Business days only

    date_decision = {}
    current_holding = 500

    try:
        auth_response = auth()
        print(f"Authentication successful: {auth_response}")
    except Exception as e:
        print(f"Error during authentication: {e}")

    try:
        workspace_name = "My New Workspace"
        workspace_slug = generateNewWorkspace(workspace_name)
        print(f"Workspace created successfully: {workspace_slug}")
    except Exception as e:
        print(f"Error during workspace creation: {e}")

    for date in daterange:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Processing date: {date_str}")
        lookback_start_date = (date - pd.Timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')
        print(f"Lookback start date: {lookback_start_date}")
        try:
            thread_slug = generateNewThread(workspace_slug)
            print(f"Thread created successfully: {thread_slug}")
        except Exception as e:
            print(f"Error during thread creation: {e}")

        
        message = message_preset.format(
            args.character_setting,
            args.ticker, lookback_start_date, date_str, date_str, current_holding,
            f"{args.ticker}, {date_str}"
        )
        # print(f"Sending message: {message}")
        try:
            chat_response = chatWithThread(workspace_slug, thread_slug, message)
            print(f"Chat response: {chat_response}")
            one_line = extract_prediction_line(chat_response, ticker=args.ticker, date=date_str)
            date_decision[date_str] = one_line
        except Exception as e:
            print(f"Error during chat: {e}")
        
        # random adjust holding from +100, 0, -100
        current_holding = max(current_holding + random.choice([-100, 0, 100]), 100)
        

        try:
            delete_response = deleteThread(workspace_slug, thread_slug)
            print(f"Thread deleted successfully: {delete_response}")
        except Exception as e:
            print(f"Error during thread deletion: {e}")

    try:
        delete_response = deleteWorkspace(workspace_slug)
        print(f"Workspace deleted successfully: {delete_response}")
    except Exception as e:
        print(f"Error during workspace deletion: {e}")

    final_results = pd.DataFrame.from_dict(date_decision, orient='index', columns=['Decision'])
    final_results.index.name = 'Date'
    final_results.reset_index(inplace=True)
    if not os.path.exists("results"):
        os.makedirs("results")
    final_results.to_csv(f"results/{workspace_slug}_decisions.csv", index=False)