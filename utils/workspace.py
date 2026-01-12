import json

import urllib3

anything_api = "1DV9A3A-SFFM1XR-QF4TYMR-HZ5X8RY"
entrypoint = "http://10.201.35.124:3001/api/v1/"

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

            raise Exception(
                f"Chat response does not contain 'textResponse': {chat_data}")
        except json.JSONDecodeError:
            raise Exception(
                f"Failed to parse chat response: {response.data.decode('utf-8')}")
    else:
        raise Exception(
            f"Chat with thread failed: {response.status} {response.data.decode('utf-8')}")
    