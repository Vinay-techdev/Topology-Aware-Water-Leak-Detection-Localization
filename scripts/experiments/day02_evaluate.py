import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

THRESHOLD = -4.0  # fixed from Scenario-13 calibration, not touched further

eval_scenarios = ["Scenario-1", "Scenario-186", "Scenario-769", "Scenario-18", "Scenario-31"]

pooled_TP = pooled_FP = pooled_FN = pooled_TN = 0

print(f"{'scenario':>15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'precision':>10} {'recall':>8} {'f1':>8}")
for name in eval_scenarios:
    bundle = load_scenario(f"data/raw/{name}")
    residuals = compute_network_avg_residuals(bundle["pressures"])
    score = scenario_anomaly_score(residuals)
    actual = bundle["labels"].astype(int)
    predicted = (score < THRESHOLD).astype(int)

    TP = int(((predicted==1)&(actual==1)).sum())
    FP = int(((predicted==1)&(actual==0)).sum())
    FN = int(((predicted==0)&(actual==1)).sum())
    TN = int(((predicted==0)&(actual==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0
    rec = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"{name:>15} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

    pooled_TP += TP; pooled_FP += FP; pooled_FN += FN; pooled_TN += TN

prec = pooled_TP/(pooled_TP+pooled_FP) if (pooled_TP+pooled_FP)>0 else 0
rec = pooled_TP/(pooled_TP+pooled_FN) if (pooled_TP+pooled_FN)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"{'POOLED':>15} {pooled_TP:>6} {pooled_FP:>6} {pooled_FN:>6} {pooled_TN:>6} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")