import re


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