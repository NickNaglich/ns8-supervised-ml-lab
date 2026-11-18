"""Hand-solved no-fit grouping/ceiling controls; not an actual-table replay."""
from itertools import product
import numpy as np
import pytest
from experiments.task_a_identifiability import (
    exact_key, group_inputs, approximate_pairs, protocol, schema, save,
)


def test_hand_solved_conflict_and_all_ties():
    keys = [exact_key(x) for x in [[0, 1], [0, 1], [2, 3], [2, 3], [2, 3]]]
    labels = ["A", "B", "A", "B", "B"]
    result = group_inputs(keys, labels, ["A", "B"])
    assert result["maximum_correct"] == 3 and result["rows"] == 5
    assert result["ceiling_fraction"] == {"numerator": 3, "denominator": 5}
    assert result["groups"][0]["members"] == [0, 1]
    assert result["groups"][0]["maximizing_labels"] == ["A", "B"]
    assert result["groups"][1]["label_counts"] == {"A": 1, "B": 2}
    # Exhaust all deterministic choices on these two exact input states.
    achieved = [sum(labels[i] == choice[0 if i < 2 else 1] for i in range(5))
                for choice in product(["A", "B"], repeat=2)]
    assert max(achieved) == result["maximum_correct"]


def test_no_rounding_or_atol_equivalence():
    X = np.array([[0.], [.75e-12], [1.5e-12]])
    keys = [exact_key(x) for x in X]
    hard = group_inputs(keys, ["A", "B", "A"], ["A", "B"])
    assert len(hard["groups"]) == 3 and hard["maximum_correct"] == 3
    approx = approximate_pairs(X, ["A", "B", "A"])
    assert [p["rows"] for p in approx] == [[0, 1], [1, 2]]
    assert all(not p["bit_exact"] for p in approx)
    assert exact_key([1.]) != exact_key([np.nextafter(1., 2.)])


def test_signed_zero_and_finite_contract():
    assert exact_key([0.]) != exact_key([-0.])
    assert exact_key([1., 2.]) != exact_key([2., 1.])
    for x in [[], [float("nan")], [float("inf")]]:
        with pytest.raises(ValueError):
            exact_key(x)


def test_raw_groups_need_not_nest_feature_groups():
    labels = ["A", "B", "B"]
    raw = group_inputs([(2, b"same"), (2, b"same"), (2, b"different")], labels, ["A", "B"])
    feat = group_inputs([exact_key([0]), exact_key([1]), exact_key([1])], labels, ["A", "B"])
    assert raw["groups"][0]["members"] == [0, 1]
    assert feat["groups"][1]["members"] == [1, 2]
    assert raw["maximum_correct"] == 2 and feat["maximum_correct"] == 3


def test_invalid_alignment_or_unknown_label_rejected():
    for keys, labels in [([], []), ([(1,)], []), ([(1,)], ["unknown"])]:
        with pytest.raises(ValueError):
            group_inputs(keys, labels, ["A", "B"])


def test_schema_budget_and_exclusive_freeze_files(tmp_path):
    names = schema(); p = protocol()
    assert len(names) == len(set(names)) == 48
    assert names[0] == "feat_hist_bin_00" and names[-1] == "feat_fft_low_ratio"
    assert len(p["N"]) * len(p["k"]) * len(p["views"]) == p["rows"] == 192
    assert sum(n*n for n in p["N"])*len(p["k"])*len(p["views"]) == p["raw_cells"] == 118784
    path = tmp_path / "freeze.json"; save(path, p)
    with pytest.raises(FileExistsError):
        save(path, {})
