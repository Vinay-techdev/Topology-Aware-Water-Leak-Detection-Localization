"""Day 04: false-positive behavior of the fixed Day 02 detector (network-average residual,
threshold -4.0, unchanged) across 5 real no-leak scenarios instead of just Scenario-1.

No threshold changes here. This only measures how one already-fixed detector behaves on
scenarios it has never been scored against.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

THRESHOLD = -4.0
DATA_DIR = Path("data/raw")

NO_LEAK_SCENARIOS = ["Scenario-1", "Scenario-289", "Scenario-489", "Scenario-694", "Scenario-844"]

# Day 02's known result for Scenario-1, used as a regression check.
DAY02_SCENARIO1_FP = 529


def evaluate(name):
    bundle = load_scenario(DATA_DIR / name)
    labels = bundle["labels"].astype(int).to_numpy()
    if labels.sum() != 0:
        raise ValueError(f"{name}: expected a no-leak scenario, but Labels.csv has {labels.sum()} positive rows")

    score = scenario_anomaly_score(compute_network_avg_residuals(bundle["pressures"]))
    predicted = (score < THRESHOLD).to_numpy()
    fp = int(predicted.sum())
    total = len(labels)
    return {
        "fp": fp,
        "total": total,
        "fp_rate": fp / total,
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_min": float(score.min()),
    }


def main():
    results = {name: evaluate(name) for name in NO_LEAK_SCENARIOS}

    got = results["Scenario-1"]["fp"]
    if got != DAY02_SCENARIO1_FP:
        raise SystemExit(f"REGRESSION MISMATCH Scenario-1: got FP={got}, Day 02 had {DAY02_SCENARIO1_FP}")
    print(f"Regression check passed: Scenario-1 reproduces Day 02's FP={DAY02_SCENARIO1_FP}.\n")

    print(f"{'scenario':>15} {'FP':>6} {'total':>7} {'fp_rate':>9} {'score_mean':>11} {'score_std':>10} {'score_min':>10}")
    for name in NO_LEAK_SCENARIOS:
        r = results[name]
        print(f"{name:>15} {r['fp']:>6} {r['total']:>7} {r['fp_rate']:>9.4f} "
              f"{r['score_mean']:>11.3f} {r['score_std']:>10.3f} {r['score_min']:>10.3f}")

    rates = [results[n]["fp_rate"] for n in NO_LEAK_SCENARIOS]
    n = len(rates)
    mean_rate = sum(rates) / n
    var = sum((x - mean_rate) ** 2 for x in rates) / (n - 1)
    std_rate = var ** 0.5
    print(f"\n{'mean':>15} {'':>6} {'':>7} {mean_rate:>9.4f}")
    print(f"{'std (n-1)':>15} {'':>6} {'':>7} {std_rate:>9.4f}")
    print(f"{'min':>15} {'':>6} {'':>7} {min(rates):>9.4f}")
    print(f"{'max':>15} {'':>6} {'':>7} {max(rates):>9.4f}")


if __name__ == "__main__":
    main()