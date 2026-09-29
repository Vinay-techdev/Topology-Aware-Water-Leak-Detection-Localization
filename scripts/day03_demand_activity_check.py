import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals

scenarios = ["Scenario-186", "Scenario-769", "Scenario-13", "Scenario-18", "Scenario-31"]


def separation(signal, mask, ref_mask):
    """(median over mask - median over reference) / std over reference."""
    part = signal[mask]
    if len(part) == 0:
        return float("nan")
    ref = signal[ref_mask]
    return (part.median() - ref.median()) / ref.std()


header_shown = False
print(f"{'scenario':>13} {'node':>4} {'type':>9} {'area':>9} {'win_len':>7} {'lbl1_in_win':>11} "
      f"{'nz_in_win':>9} {'nz_outside':>10} {'sep_window':>10} {'sep_active':>10}")

for name in scenarios:
    bundle = load_scenario(f"data/raw/{name}")
    residuals = compute_network_avg_residuals(bundle["pressures"])
    labels = bundle["labels"].astype(int)
    noleak = (labels == 0).values

    for leak in bundle["leaks"]:
        node = leak["node"]
        demand_df = pd.read_csv(f"data/raw/{name}/Leaks/Leak_{node}_demand.csv")

        if not header_shown:
            print("\n[demand file check] columns:", list(demand_df.columns), "rows:", len(demand_df))
            print(demand_df.head(3).to_string(), "\n")
            header_shown = True

        assert len(demand_df) == len(residuals), f"{name} node {node}: demand rows != timesteps"
        demand = pd.Series(demand_df.iloc[:, 1].values, index=residuals.index)

        start, end = leak["Leak Start (timestamp)"], leak["Leak End (timestamp)"]
        window = (residuals.index >= start) & (residuals.index <= end)
        nonzero = (demand != 0).values
        active = window & nonzero

        s = residuals[node]
        print(f"{name:>13} {node:>4} {str(leak['Leak Type']):>9} {float(leak['Leak Area']):>9.5f} "
              f"{int(window.sum()):>7} {int(labels[window].sum()):>11} "
              f"{int((window & nonzero).sum()):>9} {int((~window & nonzero).sum()):>10} "
              f"{separation(s, window, noleak):>10.2f} {separation(s, active, noleak):>10.2f}")