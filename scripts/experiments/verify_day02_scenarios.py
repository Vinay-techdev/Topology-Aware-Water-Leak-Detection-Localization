import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario

# What the audit CSV already told us to expect for each scenario
expected = {
    "Scenario-1":   {"num_leaks": 0, "nodes": []},
    "Scenario-13":  {"num_leaks": 2, "nodes": ["21", "30"]},
    "Scenario-186": {"num_leaks": 1, "nodes": ["6"]},
    "Scenario-769": {"num_leaks": 1, "nodes": ["9"]},
    "Scenario-18":  {"num_leaks": 2, "nodes": ["13", "29"]},
    "Scenario-31":  {"num_leaks": 2, "nodes": ["28", "8"]},
}

for name, exp in expected.items():
    bundle = load_scenario(f"data/raw/{name}")
    actual_nodes = sorted(leak["node"] for leak in bundle["leaks"])
    exp_nodes = sorted(exp["nodes"])

    ok = (len(bundle["leaks"]) == exp["num_leaks"]) and (actual_nodes == exp_nodes)

    print(f"{name:15} pressures={bundle['pressures'].shape}  "
          f"leaks_found={len(bundle['leaks'])} (expected {exp['num_leaks']})  "
          f"nodes={actual_nodes} (expected {exp_nodes})  "
          f"-> {'MATCH' if ok else 'MISMATCH !!'}")