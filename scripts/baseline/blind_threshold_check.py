"""Day 06: a per-scenario threshold computed BLINDLY - from the scenario's own full score
distribution, without using Labels.csv to pick which timesteps are "normal". Labels.csv is
used only afterward, to score the resulting predictions.

This is the direct test of the bootstrapping problem named at the end of Day 05: can a
per-scenario relative threshold work without already knowing which parts are leak-free?
Expected, stated before running on real data: this should work reasonably where leaks are
a minority of the scenario, and break down where leak-positive time dominates the scenario
(Scenario-18 is flagged in advance as the case to watch, based on Day 02's own counts:
15751 of 17520 rows, about 90%, are label-positive there).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

DATA_DIR = Path("data/raw")
SCENARIOS = ["Scenario-1", "Scenario-13", "Scenario-186", "Scenario-769", "Scenario-18", "Scenario-31"]
TARGET_FP_RATE = 0.01

# Day 02 fixed-threshold (-4.0) confusion counts (TP, FP, FN, TN) vs Labels.csv, for comparison.
DAY02_LABEL_COUNTS = {
    "Scenario-1": (0, 529, 0, 16991),
    "Scenario-13": (5803, 32, 568, 11117),
    "Scenario-186": (0, 0, 2700, 14820),
    "Scenario-769": (160, 0, 2466, 14894),
    "Scenario-18": (3578, 0, 12173, 1769),
    "Scenario-31": (521, 0, 5060, 11939),
}


def confusion(pred, truth):
    return (int((pred & truth).sum()), int((pred & ~truth).sum()),
            int((~pred & truth).sum()), int((~pred & ~truth).sum()))


def ratio(n, d):
    return n / d if d > 0 else None


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def evaluate(name):
    bundle = load_scenario(DATA_DIR / name)
    labels = bundle["labels"].astype(int).to_numpy().astype(bool)

    score = scenario_anomaly_score(compute_network_avg_residuals(bundle["pressures"])).to_numpy()
    threshold = np.percentile(score, TARGET_FP_RATE * 100)  # blind: uses ALL of score, not filtered by labels
    predicted = score < threshold

    tp, fp, fn, tn = confusion(predicted, labels)
    return {
        "threshold": float(threshold),
        "prevalence": float(labels.mean()),
        "cm": (tp, fp, fn, tn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }


def main():
    results = {name: evaluate(name) for name in SCENARIOS}

    print(f"Blind per-scenario threshold, target rate={TARGET_FP_RATE:.4f}\n")
    print(f"{'scenario':>13} {'prevalence':>10} {'threshold':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} "
          f"{'precision':>9} {'recall':>7}")
    for name in SCENARIOS:
        r = results[name]
        tp, fp, fn, tn = r["cm"]
        print(f"{name:>13} {r['prevalence']:>10.3f} {r['threshold']:>10.3f} {tp:>6} {fp:>6} {fn:>6} {tn:>6} "
              f"{fmt(r['precision']):>9} {fmt(r['recall']):>7}")

    print(f"\n{'scenario':>13} {'Day02 TP':>9} {'Day02 FP':>9} {'Day02 FN':>9} {'Day02 recall':>13} "
          f"{'Day06 recall':>13}")
    for name in SCENARIOS:
        d2tp, d2fp, d2fn, d2tn = DAY02_LABEL_COUNTS[name]
        d2recall = ratio(d2tp, d2tp + d2fn)
        d6recall = results[name]["recall"]
        print(f"{name:>13} {d2tp:>9} {d2fp:>9} {d2fn:>9} {fmt(d2recall):>13} {fmt(d6recall):>13}")


if __name__ == "__main__":
    main()