import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aquaguard.data.loaders import load_scenario
from aquaguard.features.residuals import compute_network_avg_residuals
from aquaguard.evaluation.scenario_score import scenario_anomaly_score

scenarios = ["Scenario-186", "Scenario-769", "Scenario-13", "Scenario-18", "Scenario-31"]


def separation(signal, leak_mask, noleak_mask):
    """(median during leak - median during no-leak) / std during no-leak."""
    leak_part = signal[leak_mask]
    ref_part = signal[noleak_mask]
    return (leak_part.median() - ref_part.median()) / ref_part.std()


print(f"{'scenario':>13} {'signal':>16} {'leak_median':>12} {'noleak_median':>14} {'noleak_std':>11} {'separation':>11}")
for name in scenarios:
    bundle = load_scenario(f"data/raw/{name}")
    residuals = compute_network_avg_residuals(bundle["pressures"])
    net_min = scenario_anomaly_score(residuals)
    labels = bundle["labels"].astype(int)
    noleak_mask = labels == 0

    # Reference row: the Day 02 network-minimum score, over all Label==1 timesteps
    leak_mask_all = labels == 1
    print(f"{name:>13} {'network-min':>16} {net_min[leak_mask_all].median():>12.3f} "
          f"{net_min[noleak_mask].median():>14.3f} {net_min[noleak_mask].std():>11.3f} "
          f"{separation(net_min, leak_mask_all, noleak_mask):>11.2f}")

    # One row per leaking node: that node's own residual over its own leak window
    for leak in bundle["leaks"]:
        node = leak["node"]
        start, end = leak["Leak Start (timestamp)"], leak["Leak End (timestamp)"]
        own_window = (residuals.index >= start) & (residuals.index <= end)
        s = residuals[node]
        print(f"{name:>13} {'node ' + node + ' own':>16} {s[own_window].median():>12.3f} "
              f"{s[noleak_mask].median():>14.3f} {s[noleak_mask].std():>11.3f} "
              f"{separation(s, own_window, noleak_mask):>11.2f}")