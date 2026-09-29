import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario

scenarios = ["Scenario-186", "Scenario-769", "Scenario-13", "Scenario-18", "Scenario-31"]


def read_indexed(path, value_col):
    """Read an Index,<value_col> CSV into a Series keyed by the file's own 1-based Index."""
    df = pd.read_csv(path)
    assert list(df.columns[:2]) == ["Index", value_col], f"{path}: unexpected columns {list(df.columns)}"
    assert (df["Index"].values == np.arange(1, len(df) + 1)).all(), f"{path}: Index is not 1..N"
    return pd.Series(df[value_col].values, index=df["Index"].values)


def show_boundary(title, center, labels, flow):
    print(f"      {title} (metadata index = {center})")
    print(f"      {'index':>7} {'label':>6} {'flow':>12}")
    for k in (center - 1, center, center + 1):
        if k in labels.index:
            print(f"      {k:>7} {labels[k]:>6.0f} {flow[k]:>12.4f}")


for name in scenarios:
    folder = Path(f"data/raw/{name}")
    bundle = load_scenario(folder)
    labels = read_indexed(folder / "Labels.csv", "Label")

    print("=" * 78)
    print(name)
    union = set()
    for leak in bundle["leaks"]:
        node = leak["node"]
        flow = read_indexed(folder / "Leaks" / f"Leak_{node}_demand.csv", "Value")
        s, e, p = int(leak["Leak Start"]), int(leak["Leak End"]), int(leak["Peak Time"])
        union.update(range(s, e + 1))

        nz = flow[flow != 0].index
        zero_inside = [int(i) for i in flow.index if s <= i <= e and flow[i] == 0]
        print(f"\n  Leak node {node} ({leak['Leak Type']}): metadata start={s}  end={e}  peak={p}")
        print(f"    nonzero flow: first={nz.min()}  last={nz.max()}  count={len(nz)}")
        print(f"    last nonzero flow minus peak index = {nz.max() - p}")
        print(f"    zero-flow indices inside [start, end]: count={len(zero_inside)}  first few={zero_inside[:5]}")
        show_boundary("START boundary", s, labels, flow)
        show_boundary("END boundary", e, labels, flow)

    lab1 = set(int(i) for i in labels[labels == 1].index)
    print(f"\n  Scenario-level (Labels.csv vs union of metadata windows, inclusive both ends):")
    print(f"    Label==1 count={len(lab1)}  first={min(lab1)}  last={max(lab1)}")
    print(f"    union of windows size={len(union)}  first={min(union)}  last={max(union)}")
    only_union = sorted(union - lab1)
    only_label = sorted(lab1 - union)
    print(f"    in windows but Label==0: count={len(only_union)}  indices(first 5)={only_union[:5]}")
    print(f"    Label==1 but outside windows: count={len(only_label)}  indices(first 5)={only_label[:5]}")