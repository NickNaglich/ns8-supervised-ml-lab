"""Storage/value/version boundaries using fake callbacks; zero native extracts."""
import numpy as np
import pytest
from experiments import layout_stability as trial


def test_three_controls_keep_values_dtype_and_real_stride_intervention():
    base = np.arange(16, dtype=np.int64).reshape(4, 4)
    before = trial.signature(base)
    arrays = trial.controls(base)
    assert arrays["C"].flags.c_contiguous
    assert arrays["F"].flags.f_contiguous
    assert arrays["negative_stride"].strides[0] < 0
    for grid in arrays.values():
        assert trial.signature(grid) == before
        np.testing.assert_array_equal(grid, base)
    assert trial.signature(base) == before


def test_wrapper_canonicalizes_without_mutation_or_metadata_in_X(monkeypatch):
    base = np.arange(16, dtype=np.int64).reshape(4, 4)
    observed = []
    def fake(grid, hist_bins):
        assert grid.dtype == np.dtype("int64") and grid.flags.c_contiguous
        assert hist_bins == 16
        observed.append(trial.signature(grid))
        return {name: float(i) for i, name in enumerate(trial.schema())}
    monkeypatch.setattr(trial, "native_extract", fake)
    for grid in trial.controls(base).values():
        before = trial.signature(grid); strides = grid.strides
        result = trial.c_order_features(grid)
        assert list(result) == trial.schema() and len(result) == 48
        assert trial.signature(grid) == before and grid.strides == strides
        assert "feature_variant" not in result
    assert len(observed) == 3 and all(x == observed[0] for x in observed)


@pytest.mark.parametrize("grid", [np.ones((4, 4), dtype=np.float64), np.ones((4, 4), dtype=np.int32), np.ones((2, 3), dtype=np.int64)])
def test_wrapper_refuses_dtype_or_shape_coercion(monkeypatch, grid):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before extraction")
    monkeypatch.setattr(trial, "native_extract", forbidden)
    with pytest.raises(ValueError):
        trial.c_order_features(grid)


def test_exact_identity_preserves_one_bit_changes_and_signed_zero():
    assert trial.exact_key([1.]) != trial.exact_key([np.nextafter(1., 2.)])
    assert trial.exact_key([0.]) != trial.exact_key([-0.])
