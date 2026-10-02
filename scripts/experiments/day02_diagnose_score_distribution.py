import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

all_scenarios = ["Scenario-13", "Scenario-1", "Scenario-186", "Scenario-769", "Scenario-18", "Scenario-31"]

print(f"{'scenario':>15} {'group':>8} {'count':>7} {'mean':>8} {'median':>8} {'std':>8} {'min':>8}")
for name in all_scenarios:
    bundle = load_scenario(f"data/raw/{name}")
    residuals = compute_network_avg_residuals(bundle["pressures"])
    score = scenario_anomaly_score(residuals)
    actual = bundle["labels"].astype(int)

    for label_value, group_name in [(0, "no-leak"), (1, "leak")]:
        subset = score[actual == label_value]
        if len(subset) == 0:
            print(f"{name:>15} {group_name:>8} {'(none)':>7}")
            continue
        print(f"{name:>15} {group_name:>8} {len(subset):>7} {subset.mean():>8.3f} "
              f"{subset.median():>8.3f} {subset.std():>8.3f} {subset.min():>8.3f}")