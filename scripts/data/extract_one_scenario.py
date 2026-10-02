import zipfile
import yaml
import re
from pathlib import Path

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

zip_path = Path(config["data"]["leakdb_root"]) / "Hanoi_CMH_1000scenarios.zip"

# Day 02 scenario set (see scenario-selection reasoning: 1 no-leak, 1 calibration-only,
# 4 evaluation scenarios spanning leak-count / leak-type / duration)
scenario_names = ["Scenario-289", "Scenario-489", "Scenario-694", "Scenario-844"]

with zipfile.ZipFile(zip_path, "r") as zf:
    all_names = zf.namelist()

    for scenario_name in scenario_names:
        output_dir = Path("data/raw") / scenario_name
        if output_dir.exists():
            print(f"{scenario_name} already extracted, skipping.")
            continue

        pattern = re.compile(rf"{scenario_name}/(.+)$")
        matches = [(name, pattern.search(name).group(1)) for name in all_names if pattern.search(name) and not name.endswith("/")]

        print(f"Extracting {scenario_name}: {len(matches)} files...")
        for zip_internal_path, relative_path in matches:
            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zip_internal_path) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

print("Done.")