import pandas as pd
import matplotlib
matplotlib.use("Agg")  # needed if running from terminal without a display window
import matplotlib.pyplot as plt

n19 = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_19.csv", parse_dates=["Timestamp"])
n5 = pd.read_csv("LeakDB/Hanoi_CMH/Scenario-1/Pressures/Node_5.csv", parse_dates=["Timestamp"])

residual = n19["Value"] - n5["Value"]

# These three timestamps come from Leaks/Leak_19_info.csv
leak_start = pd.Timestamp("2019-07-02 09:30:00")
leak_end = pd.Timestamp("2019-07-06 00:30:00")
peak_time = pd.Timestamp("2019-07-04 20:00:00")

plt.figure(figsize=(11, 4))
plt.plot(n19["Timestamp"], residual, color="purple", label="Node19 - Node5 (residual)")
plt.axvline(leak_start, color="red", linestyle="--", label="Leak Start")
plt.axvline(leak_end, color="black", linestyle="--", label="Leak End")
plt.axvline(peak_time, color="orange", linestyle=":", label="Leak Peak Time")
plt.axhline(0, color="gray", linewidth=0.8)
plt.xlabel("Time")
plt.ylabel("Pressure difference (m)")
plt.title("Residual: Node19 (leaking) minus Node5 (healthy) - Scenario 1")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
print("Saved residual_plot.png")