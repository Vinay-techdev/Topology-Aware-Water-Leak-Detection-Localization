import pandas as pd

# Load the two nodes' pressure data and the ground-truth labels
n19 = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_19.csv", parse_dates=["Timestamp"])
n5 = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_5.csv", parse_dates=["Timestamp"])
labels = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Labels.csv", parse_dates=["Timestamp"])

# Compute the residual signal (leaking node minus healthy reference node)
residual = n19["Value"] - n5["Value"]

# --- Change this value and re-run to see the precision/recall trade-off ---
threshold = -3.0

# Predict 1 (leak) whenever residual drops below the threshold
predicted = (residual < threshold).astype(int)
actual = labels["Label"].astype(int)

TP = ((predicted == 1) & (actual == 1)).sum()
FP = ((predicted == 1) & (actual == 0)).sum()
FN = ((predicted == 0) & (actual == 1)).sum()
TN = ((predicted == 0) & (actual == 0)).sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Threshold: {threshold}")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")