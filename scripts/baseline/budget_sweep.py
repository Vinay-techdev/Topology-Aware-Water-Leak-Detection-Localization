"""Day 07: sweep the blind per-scenario percentile threshold across several target rates,
instead of the single fixed 1% used in Day 06. Still fully blind - Labels.csv is never used
to pick the threshold, only afterward to score it.

Directly repairs a methodology error found at the end of Day 06: at one fixed target rate,
"recall" mostly measured leak duration (flagged-point budget vs leak size), not detector
quality. Sweeping the rate separates those two things.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

DATA_DIR = Path("data/raw")
SCENARIOS = ["Scenario-13", "Scenario-186", "Scenario-769", "Scenario-18", "Scenario-31"]
TARGET_RATES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]


def confusion(pred, truth):
    return (int((pred & truth).sum()), int((pred & ~truth).sum()),
            int((~pred & truth).sum()), int((~pred & ~truth).sum()))


def ratio(n, d):
    return n / d if d > 0 else None


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def evaluate_at_rate(score, labels, rate):
    threshold = np.percentile(score, rate * 100)
    predicted = score < threshold
    tp, fp, fn, tn = confusion(predicted, labels)
    return ratio(tp, tp + fp), ratio(tp, tp + fn)


def main():
    print(f"{'scenario':>13} {'prevalence':>10}", end="")
    for rate in TARGET_RATES:
        print(f" | budget={rate:.0%}".rjust(9), end="")
    print()
    print(f"{'':>13} {'':>10}", end="")
    for _ in TARGET_RATES:
        print(f"{'prec/rec':>9}", end="")
    print()

    for name in SCENARIOS:
        bundle = load_scenario(DATA_DIR / name)
        labels = bundle["labels"].astype(int).to_numpy().astype(bool)
        score = scenario_anomaly_score(compute_network_avg_residuals(bundle["pressures"])).to_numpy()
        prevalence = labels.mean()

        print(f"{name:>13} {prevalence:>10.3f}", end="")
        for rate in TARGET_RATES:
            prec, rec = evaluate_at_rate(score, labels, rate)
            print(f" {fmt(prec)[:4]}/{fmt(rec)[:4]}".rjust(9), end="")
        print()


if __name__ == "__main__":
    main()