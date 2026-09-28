import sys
from pathlib import Path

# Add the "src" folder to Python's search path, so "aquaguard.xxx" imports resolve correctly
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd


def compute_network_avg_residuals(pressures: pd.DataFrame, reservoir_node: str = "1") -> pd.DataFrame:
    junction_columns = [c for c in pressures.columns if c != reservoir_node]
    network_avg = pressures[junction_columns].mean(axis=1)
    residuals = pressures[junction_columns].sub(network_avg, axis=0)
    return residuals


if __name__ == "__main__":
    from aquaguard.data.loaders import load_scenario

    bundle = load_scenario("data/raw/Scenario-13")
    residuals = compute_network_avg_residuals(bundle["pressures"])

    print("Residuals shape:", residuals.shape)
    print()
    for leak in bundle["leaks"]:
        node = leak["node"]
        leak_start = leak["Leak Start (timestamp)"]
        leak_end = leak["Leak End (timestamp)"]

        node_residual = residuals[node]

        before = node_residual[node_residual.index < leak_start]
        during = node_residual[(node_residual.index >= leak_start) & (node_residual.index <= leak_end)]
        after = node_residual[node_residual.index > leak_end]

        print(f"--- Node {node} ({leak['Leak Type']} leak) ---")
        print(f"Leak window: {leak_start} to {leak_end}")
        print(f"BEFORE leak  -> mean: {before.mean():.3f}, std: {before.std():.3f}, min: {before.min():.3f}")
        print(f"DURING leak  -> mean: {during.mean():.3f}, std: {during.std():.3f}, min: {during.min():.3f}")
        print(f"AFTER leak   -> mean: {after.mean():.3f}, std: {after.std():.3f}, min: {after.min():.3f}")
        print()