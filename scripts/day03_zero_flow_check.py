"""Day 03: for each incipient leak, compare the leaking node's own residual on rows where
Label==1 but that leak's own demand is zero, against Label==0 rows.

If those zero-flow rows look statistically like no-leak rows (separation near 0), that
supports treating a leak's own demand file as "is it physically leaking right now" rather
than relying on Labels.csv's window alone. This does not touch the fixed Day 02 detector
or its threshold - it only examines the node's own residual signal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals

DATA_DIR = Path("data/raw")

# (scenario, leak node) pairs identified as incipient in the Day 03 boundary check.
INCIPIENT_LEAKS = [
    ("Scenario-769", "9"),
    ("Scenario-31", "28"),
    ("Scenario-31", "8"),
]


def read_demand(scenario_folder: Path, node: str) -> np.ndarray:
    path = scenario_folder / "Leaks" / f"Leak_{node}_demand.csv"
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["Index", "Value"]:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)}")
    return df["Value"].to_numpy()


def separation(signal: np.ndarray, mask: np.ndarray, ref_mask: np.ndarray):
    """(median over mask - median over reference) / std over reference. None if mask is empty."""
    part = signal[mask]
    if len(part) == 0:
        return None
    ref = signal[ref_mask]
    return (np.median(part) - np.median(ref)) / np.std(ref, ddof=1)


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def main():
    print(f"{'scenario':>13} {'node':>4} {'own_window':>10} {'zero_in_win':>11} {'active_in_win':>13} "
          f"{'sep_zero':>9} {'sep_active':>10}")

    for scenario_name, node in INCIPIENT_LEAKS:
        folder = DATA_DIR / scenario_name
        bundle = load_scenario(folder)
        residuals = compute_network_avg_residuals(bundle["pressures"])

        leak = next(lk for lk in bundle["leaks"] if lk["node"] == node)
        start, end = leak["Leak Start (timestamp)"], leak["Leak End (timestamp)"]

        demand = read_demand(folder, node)
        if len(demand) != len(residuals):
            raise ValueError(f"{scenario_name} node {node}: demand has {len(demand)} rows, "
                              f"residuals have {len(residuals)}")

        own_window = np.asarray((residuals.index >= start) & (residuals.index <= end))
        nonzero = demand != 0
        zero_in_window = own_window & ~nonzero
        active_in_window = own_window & nonzero

        labels = bundle["labels"].astype(int).to_numpy()
        noleak_mask = labels == 0

        s = residuals[node].to_numpy()
        sep_zero = separation(s, zero_in_window, noleak_mask)
        sep_active = separation(s, active_in_window, noleak_mask)

        print(f"{scenario_name:>13} {node:>4} {int(own_window.sum()):>10} "
              f"{int(zero_in_window.sum()):>11} {int(active_in_window.sum()):>13} "
              f"{fmt(sep_zero):>9} {fmt(sep_active):>10}")


if __name__ == "__main__":
    main()