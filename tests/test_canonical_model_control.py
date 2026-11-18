"""Meaningful fake-only controls; zero native extractions/classifier/scaler fits."""
import numpy as np
import pytest
from experiments.canonical_model_control import frame, matrix, qualification, raw_overlap, fit_once


def test_exact_qualification_checks_all_dimensions_and_lengths():
    X = np.zeros((2, 48)); expected = [[float(v).hex() for v in row] for row in X]
    assert qualification(X, expected)["match"]
    X[1, 47] = np.nextafter(0., 1.)
    result = qualification(X, expected)
    assert not result["match"] and result["different"] == [{"row": 1, "columns": [47]}]
    assert not qualification(X[:1], expected)["match"]


def test_raw_origin_crossing_rejectable_with_fixed_membership():
    groups = {"groups": [{"group": 0, "members": [0, 1]}, {"group": 1, "members": [2]}]}
    assert raw_overlap(groups, [0, 2], [1]) == [0]
    assert raw_overlap(groups, [0, 1], [2]) == []


def test_metadata_not_X_and_schema_order_explicit():
    df = frame([{"row": 0, "N": 16, "k": 1, "target": "TLF", "feature_pipeline_id": "meta"}], [[1., 2.]], ["feat_a", "feat_b"])
    X = matrix(df, ["feat_a", "feat_b"])
    assert X.tolist() == [[1., 2.]] and X.flags.c_contiguous
    with pytest.raises(ValueError): matrix(df, ["feat_b", "feat_a"])


def test_same_fixed_indices_pass_only_training_rows_to_each_fake_model():
    class Fake:
        def fit(self, X, y): self.seen = (X.copy(), y.copy()); return self
        def predict(self, X): self.test = X.copy(); return np.array(["A"]*len(X))
        def decision_function(self, X): return np.zeros((len(X), 2))
    y = np.array(["A", "B", "A", "B"]); train = np.array([0, 2]); test = np.array([1, 3])
    for X in [np.arange(8).reshape(4, 2), np.arange(8).reshape(4, 2)+1]:
        model = Fake(); fit_once(model, X, y, train, test)
        np.testing.assert_array_equal(model.seen[0], X[train]); np.testing.assert_array_equal(model.seen[1], y[train]); np.testing.assert_array_equal(model.test, X[test])
