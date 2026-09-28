import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load data for Node 19 (leaking), Node 5 (normal), and Node 1 (reservoir)
df_19 = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_19.csv", parse_dates=["Timestamp"])
df_5  = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_5.csv", parse_dates=["Timestamp"])
df_1  = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_1.csv", parse_dates=["Timestamp"])

# 2. Set leak start time
leak_start = pd.Timestamp("2019-07-02 09:30:00")

# 3. Create the comparison plot
plt.figure(figsize=(12, 5))

plt.plot(df_19["Timestamp"], df_19["Value"], label="Node 19 (Leaking)", color="tab:blue")
plt.plot(df_5["Timestamp"], df_5["Value"], label="Node 5 (No Leak)", color="tab:green", alpha=0.8)
plt.plot(df_1["Timestamp"], df_1["Value"], label="Node 1 (Reservoir)", color="tab:gray", linestyle=":")

# Highlight the leak event start
plt.axvline(leak_start, color="red", linestyle="--", linewidth=1.5, label="Leak Start")

plt.xlabel("Time")
plt.ylabel("Pressure (m)")
plt.title("Pressure Comparison across Nodes — Scenario 1")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

# Save the plot
plt.savefig("node_comparison_plot.png", dpi=150)
print("Plot saved as node_comparison_plot.png")