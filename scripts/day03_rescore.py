"""Day 03: re-score the fixed Day 02 detector against two ground-truth definitions.

The threshold is the one fixed on Scenario-13 in Day 02. It is NOT re-tuned here.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score
from aquaguard.evaluation.ground_truth import load_ground_truth_masks

THRESHOLD = -4.0
DATA_DIR = Path("data/raw")

HELD_OUT = ["Scenario-1", "Scenario-186", "Scenario-769", "Scenario-18", "Scenario-31"]
CALIBRATION = "Scenario-13"  # reported separately, never pooled with the held-out set

# Day 02 confusion counts (TP, FP, FN, TN) against Labels.csv at THRESHOLD.
# Used as a regression check: this script must reproduce them exactly.
DAY02_LABEL_COUNTS = {
    "Scenario-1": (0, 529, 0, 16991),
    "Scenario-186": (0, 0, 2700, 14820),
    "Scenario-769": (160, 0, 2466, 14894),
    "Scenario-18": (3578, 0, 12173, 1769),
    "Scenario-31": (521, 0, 5060, 11939),
    "Scenario-13": (5803, 32, 568, 11117),
}


def confusion(pred, truth):
    return (int((pred & truth).sum()), int((pred & ~truth).sum()),
            int((~pred & truth).sum()), int((~pred & ~truth).sum()))


def ratio(num, den):
    return num / den if den > 0 else None


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def evaluate_scenario(name):
    bundle = load_scenario(DATA_DIR / name)
    score = scenario_anomaly_score(compute_network_avg_residuals(bundle["pressures"]))
    truth = load_ground_truth_masks(DATA_DIR / name)
    if len(score) != len(truth):
        raise ValueError(f"{name}: {len(score)} scores but {len(truth)} truth rows")
    pred = (score < THRESHOLD).to_numpy()
    label = truth["label"].to_numpy()
    flow = truth["flow_active"].to_numpy()
    return {
        "label_cm": confusion(pred, label),
        "flow_cm": confusion(pred, flow),
        "n_label": int(label.sum()),
        "n_flow": int(flow.sum()),
        "flow_outside_label": int((flow & ~label).sum()),
        "pred_total": int(pred.sum()),
        "pred_on_flow": int((pred & flow).sum()),
        "pred_label_no_flow": int((pred & label & ~flow).sum()),
        "pred_outside_label": int((pred & ~label & ~flow).sum()),
    }


def add_up(results, names, key):
    values = [results[n][key] for n in names]
    return tuple(map(sum, zip(*values))) if isinstance(values[0], tuple) else sum(values)


def print_confusion_table(title, key, results):
    print(title)
    print(f"{'scenario':>22} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'precision':>10} {'recall':>8}")
    rows = [(n, results[n][key]) for n in HELD_OUT] + [("POOLED (held-out)", add_up(results, HELD_OUT, key))]
    rows.append((f"{CALIBRATION} (calib.)", results[CALIBRATION][key]))
    for name, (tp, fp, fn, tn) in rows:
        print(f"{name:>22} {tp:>6} {fp:>6} {fn:>6} {tn:>6} {fmt(ratio(tp, tp + fp)):>10} {fmt(ratio(tp, tp + fn)):>8}")
    print()


def main():
    results = {name: evaluate_scenario(name) for name in HELD_OUT + [CALIBRATION]}

    for name, expected in DAY02_LABEL_COUNTS.items():
        got = results[name]["label_cm"]
        if got != expected:
            raise SystemExit(f"REGRESSION MISMATCH {name}: got {got}, Day 02 had {expected}")
    print("Regression check passed: Labels.csv counts reproduce Day 02 for all 6 scenarios.\n")

    print("A. Mask sizes (rows)")
    print(f"{'scenario':>22} {'label==1':>9} {'flow_active':>12} {'flow_outside_label':>19}")
    for n in HELD_OUT + [CALIBRATION]:
        r = results[n]
        print(f"{n:>22} {r['n_label']:>9} {r['n_flow']:>12} {r['flow_outside_label']:>19}")
    print()

    print_confusion_table("B. Confusion vs Labels.csv (should equal Day 02)", "label_cm", results)
    print_confusion_table("C. Confusion vs flow-active mask", "flow_cm", results)

    print("D. Where the predicted positives fall")
    print(f"{'scenario':>22} {'predicted':>10} {'on_flow':>8} {'label_no_flow':>14} {'outside_label':>14}")
    keys = ("pred_total", "pred_on_flow", "pred_label_no_flow", "pred_outside_label")
    for n in HELD_OUT:
        print(f"{n:>22} " + " ".join(f"{results[n][k]:>{w}}" for k, w in zip(keys, (10, 8, 14, 14))))
    pooled = [add_up(results, HELD_OUT, k) for k in keys]
    print(f"{'POOLED (held-out)':>22} " + " ".join(f"{v:>{w}}" for v, w in zip(pooled, (10, 8, 14, 14))))
    print(f"{CALIBRATION + ' (calib.)':>22} " + " ".join(f"{results[CALIBRATION][k]:>{w}}" for k, w in zip(keys, (10, 8, 14, 14))))


if __name__ == "__main__":
    main()