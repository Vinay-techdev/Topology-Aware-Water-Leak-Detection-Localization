import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

bundle = load_scenario("data/raw/Scenario-13")
residuals = compute_network_avg_residuals(bundle["pressures"])
score = scenario_anomaly_score(residuals)
actual = bundle["labels"].astype(int)

print(f"{'threshold':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'precision':>10} {'recall':>8} {'f1':>8}")
for threshold in [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0]:
    predicted = (score < threshold).astype(int)
    TP = int(((predicted==1)&(actual==1)).sum())
    FP = int(((predicted==1)&(actual==0)).sum())
    FN = int(((predicted==0)&(actual==1)).sum())
    TN = int(((predicted==0)&(actual==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0
    rec = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"{threshold:>10} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")