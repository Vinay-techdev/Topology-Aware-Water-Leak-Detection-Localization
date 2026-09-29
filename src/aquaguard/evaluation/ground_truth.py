from pathlib import Path

import numpy as np
import pandas as pd


def _read_indexed(path: Path, value_col: str) -> np.ndarray:
    """Read an `Index,<value_col>` CSV; require Index == 1..N in order and no missing values."""
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["Index", value_col]:
        raise ValueError(f"{path}: expected columns ['Index', '{value_col}'], got {list(df.columns)}")
    if not (df["Index"].to_numpy() == np.arange(1, len(df) + 1)).all():
        raise ValueError(f"{path}: Index column is not 1..N in order")
    values = df[value_col].to_numpy()
    if pd.isna(values).any():
        raise ValueError(f"{path}: contains missing values")
    return values


def load_ground_truth_masks(scenario_folder) -> pd.DataFrame:
    """
    Two boolean ground-truth masks for one real LeakDB scenario, built directly from the
    scenario's files. No leak start/end convention from the metadata is used.

    label        Labels.csv == 1 (the official definition)
    flow_active  True where at least one Leaks/Leak_<node>_demand.csv value is nonzero.
                 Assumes a nonzero value means the leak is flowing (unverified).

    Rows follow file order; the DataFrame is indexed by the files' own 1-based Index.
    """
    folder = Path(scenario_folder)
    label = _read_indexed(folder / "Labels.csv", "Label") == 1
    flow_active = np.zeros(len(label), dtype=bool)

    for demand_file in sorted((folder / "Leaks").glob("Leak_*_demand.csv")):
        values = _read_indexed(demand_file, "Value")
        if len(values) != len(label):
            raise ValueError(f"{demand_file}: {len(values)} rows, but Labels.csv has {len(label)}")
        flow_active |= values != 0

    return pd.DataFrame(
        {"label": label, "flow_active": flow_active},
        index=np.arange(1, len(label) + 1),
    )