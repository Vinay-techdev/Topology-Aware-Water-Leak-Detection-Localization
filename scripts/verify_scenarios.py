import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario

expected_no_leak = ["Scenario-289", "Scenario-489", "Scenario-694", "Scenario-844"]

for name in expected_no_leak:
    bundle = load_scenario(f"data/raw/{name}")
    n_leaks = len(bundle["leaks"])
    ok = (bundle["pressures"].shape == (17520, 32)) and (n_leaks == 0)
    print(f"{name:15} pressures={bundle['pressures'].shape}  leaks_found={n_leaks} (expected 0)  "
          f"-> {'MATCH' if ok else 'MISMATCH !!'}")