import math

import numpy as np

from ns8lab.features import DEFAULT_HIST_BINS, extract_features
from ns8lab.grids import a_tlf


def test_extract_features_returns_expected_length_and_keys():
    grid = a_tlf(3, 1)
    feats = extract_features(grid, hist_bins=3)

    assert len(feats) == 35  # updated after adding skew/kurtosis and low-frequency energy
    assert all(key.startswith("feat_") for key in feats.keys())
    assert "feat_hist_entropy" in feats and "feat_hist_gini" in feats
    assert "feat_symmetry_horizontal" in feats and "feat_fft_high_ratio" in feats


def test_histogram_entropy_and_gini_for_uniform_counts():
    grid = a_tlf(3, 1)
    feats = extract_features(grid, hist_bins=3)
    expected_entropy = math.log2(3)
    expected_gini = 2.0 / 3.0

    assert math.isclose(feats["feat_hist_entropy"], expected_entropy, rel_tol=1e-2)
    assert math.isclose(feats["feat_hist_gini"], expected_gini, rel_tol=1e-2)
    assert math.isclose(feats["feat_hist_bin_00"], 1 / 3, rel_tol=1e-2)
    assert math.isclose(feats["feat_hist_bin_01"], 1 / 3, rel_tol=1e-2)
    assert math.isclose(feats["feat_hist_bin_02"], 1 / 3, rel_tol=1e-2)


def test_default_hist_bins_changes_feature_count():
    grid = a_tlf(4, 1)
    feats_default = extract_features(grid)
    feats_custom = extract_features(grid, hist_bins=8)
    assert len(feats_default) == len(feats_custom) + (DEFAULT_HIST_BINS - 8)
