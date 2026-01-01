import numpy as np
import pytest

from ns8lab.grids import VALID_VIEWS, a_tlf, a_trb, generate_all_views, generate_grid


def test_a_tlf_small_grid_matches_formula():
    grid = a_tlf(4, 1)
    expected = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(grid, expected)


def test_trb_orientation_is_transposed_flip_of_tlf():
    base = a_tlf(4, 1)
    trb = a_trb(4, 1)
    expected_trb = np.transpose(np.fliplr(base))
    np.testing.assert_array_equal(trb, expected_trb)


def test_all_views_exist_and_preserve_value_range():
    views = generate_all_views(4, 1)
    assert set(views.keys()) == set(VALID_VIEWS)
    base_min = min(v.min() for v in views.values())
    base_max = max(v.max() for v in views.values())
    assert base_min == 1 and base_max == 4
    # All views should be permutations of the same values
    first = list(views.values())[0].ravel()
    for grid in views.values():
        assert np.array_equal(np.sort(grid.ravel()), np.sort(first))


def test_generate_grid_is_deterministic():
    g1 = generate_grid(6, 3, "BRB")
    g2 = generate_grid(6, 3, "BRB")
    np.testing.assert_array_equal(g1, g2)


def test_k_parameter_changes_slope_and_wraps_modulo_n():
    grid = generate_grid(5, 2, "TLF")
    assert tuple(grid[0]) == (1, 3, 5, 2, 4)
    assert grid.shape == (5, 5)
    assert grid.min() == 1 and grid.max() == 5


def test_invalid_view_raises():
    with pytest.raises(ValueError):
        generate_grid(4, 1, "bad-view")
