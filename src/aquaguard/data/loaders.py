import pandas as pd
from pathlib import Path
import re


def load_all_node_csvs(folder: Path, prefix: str) -> pd.DataFrame:
    """Load every <prefix>_*.csv in a folder into one DataFrame (columns sorted numerically)."""
    files = sorted(folder.glob(f"{prefix}_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    data = {}
    for f in files:
        entity_id = f.stem.split("_")[1]
        df = pd.read_csv(f)
        data[entity_id] = df["Value"].values
    return pd.DataFrame(data)


def load_leaks(leaks_folder: Path, timestamps: pd.Series) -> list:
    """
    Parse every Leak_<node>_info.csv in a scenario's Leaks/ folder.
    Converts the raw integer Start/End/Peak indices into real timestamps
    using this scenario's own Timestamps.csv (index -> datetime).
    """
    leaks = []
    if not leaks_folder.exists():
        return leaks

    for info_file in sorted(leaks_folder.glob("Leak_*_info.csv")):
        node_id = re.match(r"Leak_(\d+)_info\.csv", info_file.name).group(1)
        df = pd.read_csv(info_file)

        fields = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            value = row.iloc[1]
            if isinstance(value, str):
                value = value.strip()
            fields[key] = value

        # Convert the raw 1-based index into a real timestamp using this scenario's own index
        for time_field in ("Leak Start", "Leak End", "Peak Time"):
            if time_field in fields:
                idx = int(fields[time_field])
                fields[time_field + " (timestamp)"] = timestamps.iloc[idx - 1]

        leaks.append({"node": node_id, **fields})

    return leaks


def load_scenario(scenario_folder) -> dict:
    """Load one full scenario into a bundle: pressures, flows, demands, labels, timestamps, leaks."""
    scenario_folder = Path(scenario_folder)

    #? Timestamps.csv error handling
    timestamps_path = scenario_folder / "Timestamps.csv"

    if not timestamps_path.exists():
        raise FileNotFoundError(
            f"{timestamps_path} not found. load_scenario() assumes the REAL LeakDB "
            f"schema (Index-based CSVs + Timestamps.csv), not the small toy/demo folder schema."
        )
    
    timestamps = pd.read_csv(timestamps_path)["Timestamp"]
    timestamps = pd.to_datetime(timestamps)

    pressures = load_all_node_csvs(scenario_folder / "Pressures", "Node")
    demands = load_all_node_csvs(scenario_folder / "Demands", "Node")
    flows = load_all_node_csvs(scenario_folder / "Flows", "Link")

    labels = pd.read_csv(scenario_folder / "Labels.csv")["Label"]

    leaks = load_leaks(scenario_folder / "Leaks", timestamps)

    for df in (pressures, demands, flows):
        df.index = timestamps
    labels.index = timestamps

    return {
        "pressures": pressures,
        "demands": demands,
        "flows": flows,
        "labels": labels,
        "leaks": leaks,
    }


if __name__ == "__main__":
    bundle = load_scenario("data/raw/Scenario-13")
    print("Pressures shape:", bundle["pressures"].shape)
    print("Flows shape:", bundle["flows"].shape)
    print("Demands shape:", bundle["demands"].shape)
    print("Labels shape:", bundle["labels"].shape)
    print("Number of leaks in this scenario:", len(bundle["leaks"]))
    print("First timestamp:", bundle["pressures"].index[0])
    print("Last timestamp: ", bundle["pressures"].index[-1])

    for leak in bundle["leaks"]:
        print(leak)