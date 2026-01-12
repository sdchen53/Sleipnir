import csv
import random

"""
From 1751472000 to 1751558340, generate random operations with timestamp,Decision,Hands, and save to a CSV file.
"""
start_ts = 1751472000
end_ts = 1751558340
num_records = 24*60

holding = 0  # initial hands
max_holding = 1000  # maximum hands
with open('results/my-new-workspace-3042_BTC.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'Decision', 'Hands'])
    ts = start_ts
    for _ in range(num_records):
        ts = ts + 60
        weights = [450, 121, 100]
        decision = random.choices([0, 1, -1], weights=weights)[0]  # 0: no operation, 1: buy, -1: sell
        hands = random.choice([100, 200, 300])

        if decision == 1:
            if holding + hands <= max_holding:
                holding += hands
            else:
                decision = 0
                hands = 0
        elif decision == -1:
            if holding - hands >= 0:
                holding -= hands
            else:
                decision = 0  # can't sell below zero, so no operation
                hands = 0

        writer.writerow([ts, decision, hands])