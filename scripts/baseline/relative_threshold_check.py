"""Day 05: does a per-scenario relative threshold control the false-positive rate,
where Day 02's single fixed threshold (-4.0) did not?

Each scenario computes its OWN cutoff from its OWN score distribution, targeting a
chosen false-positive rate (e.g. 1%). This is only valid here because all 5 scenarios
are known, in full, to be leak-free - so "this scenario's own distribution" is a clean
no-leak reference. That assumption does NOT hold for a real leak scenario, where you
don't know in advance which parts are normal. This script does not attempt to solve
that; it only tests whether the relative-threshold mechanism itself behaves as intended
when the assumption is met.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

DATA_DIR = Path("data/raw")
NO_LEAK_SCENARIOS = ["Scenario-1", "Scenario-289", "Scenario-489", "Scenario-694", "Scenario-844"]
TARGET_FP_RATE = 0.01  # flag the most extreme 1% of each scenario's own scores


def evaluate(name):
    bundle = load_scenario(DATA_DIR / name)
    labels = bundle["labels"].astype(int).to_numpy()
    if labels.sum() != 0:
        raise ValueError(f"{name}: expected a no-leak scenario, Labels.csv has {labels.sum()} positive rows")

    score = scenario_anomaly_score(compute_network_avg_residuals(bundle["pressures"])).to_numpy()
    threshold = np.percentile(score, TARGET_FP_RATE * 100)
    predicted = score < threshold
    fp = int(predicted.sum())
    return {"threshold": float(threshold), "fp": fp, "total": len(score), "fp_rate": fp / len(score)}


def main():
    results = {name: evaluate(name) for name in NO_LEAK_SCENARIOS}

    print(f"Target false-positive rate: {TARGET_FP_RATE:.4f}\n")
    print(f"{'scenario':>15} {'own_threshold':>14} {'FP':>6} {'total':>7} {'fp_rate':>9}")
    for name in NO_LEAK_SCENARIOS:
        r = results[name]
        print(f"{name:>15} {r['threshold']:>14.3f} {r['fp']:>6} {r['total']:>7} {r['fp_rate']:>9.4f}")

    rates = [results[n]["fp_rate"] for n in NO_LEAK_SCENARIOS]
    thresholds = [results[n]["threshold"] for n in NO_LEAK_SCENARIOS]
    n = len(rates)
    mean_rate = sum(rates) / n
    std_rate = (sum((x - mean_rate) ** 2 for x in rates) / (n - 1)) ** 0.5

    print(f"\n{'fp_rate  mean':>15} {mean_rate:>14.4f}")
    print(f"{'fp_rate  std':>15} {std_rate:>14.4f}")
    print(f"{'fp_rate  min':>15} {min(rates):>14.4f}")
    print(f"{'fp_rate  max':>15} {max(rates):>14.4f}")
    print(f"\n{'threshold min':>15} {min(thresholds):>14.3f}")
    print(f"{'threshold max':>15} {max(thresholds):>14.3f}")
    print("(Day 04, same 5 scenarios, ONE shared threshold of -4.0: fp_rate mean=0.0300, std=0.0511)")


if __name__ == "__main__":
    main()