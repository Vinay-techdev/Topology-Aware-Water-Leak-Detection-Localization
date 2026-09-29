import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from aquaguard.evaluation.ground_truth import load_ground_truth_masks

N = 20


def _write(path, col, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Index": range(1, len(values) + 1), col: values}).to_csv(path, index=False)


def _make_scenario(tmp_path, label_rows, leaks):
    """leaks: {node: set of 1-based rows with nonzero flow}"""
    _write(tmp_path / "Labels.csv", "Label", [1.0 if i in label_rows else 0.0 for i in range(1, N + 1)])
    for node, rows in leaks.items():
        _write(tmp_path / "Leaks" / f"Leak_{node}_demand.csv", "Value",
               [100.0 if i in rows else 0.0 for i in range(1, N + 1)])
    return tmp_path


def test_masks_match_files(tmp_path):
    folder = _make_scenario(tmp_path, set(range(6, 16)), {"7": set(range(7, 16)), "9": {10, 12}})
    gt = load_ground_truth_masks(folder)
    assert list(gt.columns) == ["label", "flow_active"]
    assert list(gt.index[:2]) == [1, 2] and len(gt) == N
    assert set(gt.index[gt["label"]]) == set(range(6, 16))
    assert set(gt.index[gt["flow_active"]]) == set(range(7, 16))  # union; {10, 12} already inside


def test_flow_union_of_two_leaks(tmp_path):
    folder = _make_scenario(tmp_path, set(range(2, 20)), {"1": {3, 4}, "2": {10, 11}})
    gt = load_ground_truth_masks(folder)
    assert set(gt.index[gt["flow_active"]]) == {3, 4, 10, 11}


def test_no_leaks_folder_gives_all_false_flow(tmp_path):
    folder = _make_scenario(tmp_path, set(), {})
    gt = load_ground_truth_masks(folder)
    assert not gt["label"].any() and not gt["flow_active"].any()


def test_bad_columns_raise(tmp_path):
    pd.DataFrame({"Timestamp": range(1, N + 1), "Label": [0.0] * N}).to_csv(tmp_path / "Labels.csv", index=False)
    with pytest.raises(ValueError, match="expected columns"):
        load_ground_truth_masks(tmp_path)


def test_length_mismatch_raises(tmp_path):
    folder = _make_scenario(tmp_path, {5}, {})
    _write(folder / "Leaks" / "Leak_3_demand.csv", "Value", [0.0] * (N - 1))
    with pytest.raises(ValueError, match="rows, but Labels.csv has"):
        load_ground_truth_masks(folder)


def test_missing_value_raises(tmp_path):
    folder = _make_scenario(tmp_path, {5}, {})
    _write(folder / "Leaks" / "Leak_3_demand.csv", "Value", [0.0] * (N - 1) + [float("nan")])
    with pytest.raises(ValueError, match="missing values"):
        load_ground_truth_masks(folder)