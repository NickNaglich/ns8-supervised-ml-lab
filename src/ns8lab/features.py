"""Feature extraction for NS8 grids."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable

import numpy as np

DEFAULT_HIST_BINS = 16


def _normalized_histogram(grid: np.ndarray, bins: int) -> Dict[str, float]:
    max_val = int(np.max(grid))
    edges = np.linspace(0.5, max_val + 0.5, bins + 1)
    counts, _ = np.histogram(grid, bins=edges)
    counts = counts.astype(float)
    probs = counts / counts.sum()

    hist_features = OrderedDict()
    for i, value in enumerate(probs):
        hist_features[f"feat_hist_bin_{i:02d}"] = float(value)
    hist_features["feat_hist_entropy"] = float(_entropy(probs))
    hist_features["feat_hist_gini"] = float(_gini(probs))
    return hist_features


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    nonzero = probs > 0
    return float(-np.sum(probs[nonzero] * np.log2(probs[nonzero] + eps)))


def _gini(probs: np.ndarray) -> float:
    return float(1.0 - np.sum(np.square(probs)))


def _row_col_stats(grid: np.ndarray) -> Dict[str, float]:
    row_means = grid.mean(axis=1)
    col_means = grid.mean(axis=0)
    row_vars = grid.var(axis=1)
    col_vars = grid.var(axis=0)

    def _skew(vec: np.ndarray) -> float:
        mu = vec.mean()
        sigma = vec.std()
        if sigma == 0:
            return 0.0
        centered = vec - mu
        return float(np.mean((centered / sigma) ** 3))

    def _kurtosis(vec: np.ndarray) -> float:
        mu = vec.mean()
        sigma = vec.std()
        if sigma == 0:
            return 0.0
        centered = vec - mu
        # Excess kurtosis (subtract 3)
        return float(np.mean((centered / sigma) ** 4) - 3.0)

    row_skew = np.array([_skew(r) for r in grid])
    col_skew = np.array([_skew(c) for c in grid.T])
    row_kurt = np.array([_kurtosis(r) for r in grid])
    col_kurt = np.array([_kurtosis(c) for c in grid.T])

    return {
        "feat_row_mean_mean": float(row_means.mean()),
        "feat_row_mean_std": float(row_means.std()),
        "feat_col_mean_mean": float(col_means.mean()),
        "feat_col_mean_std": float(col_means.std()),
        "feat_row_var_mean": float(row_vars.mean()),
        "feat_row_var_std": float(row_vars.std()),
        "feat_col_var_mean": float(col_vars.mean()),
        "feat_col_var_std": float(col_vars.std()),
        "feat_row_skew_mean": float(row_skew.mean()),
        "feat_row_skew_std": float(row_skew.std()),
        "feat_col_skew_mean": float(col_skew.mean()),
        "feat_col_skew_std": float(col_skew.std()),
        "feat_row_kurt_mean": float(row_kurt.mean()),
        "feat_row_kurt_std": float(row_kurt.std()),
        "feat_col_kurt_mean": float(col_kurt.mean()),
        "feat_col_kurt_std": float(col_kurt.std()),
        "feat_value_min": float(grid.min()),
        "feat_value_max": float(grid.max()),
        "feat_value_mean": float(grid.mean()),
        "feat_value_std": float(grid.std()),
    }


def _lag_autocorr(vec: np.ndarray, lag: int) -> float:
    if vec.size <= lag:
        return 0.0
    x = vec[:-lag]
    y = vec[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _axis_autocorr(grid: np.ndarray, axis: int, lags: Iterable[int]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    slices = [grid[i, :] if axis == 0 else grid[:, i] for i in range(grid.shape[axis])]
    for lag in lags:
        scores = [_lag_autocorr(vec, lag) for vec in slices]
        results[f"feat_autocorr_axis{axis}_lag{lag}"] = float(np.mean(scores))
    return results


def _symmetry_scores(grid: np.ndarray) -> Dict[str, float]:
    value_range = float(np.ptp(grid))
    denom = value_range if value_range != 0 else 1.0

    horizontal = 1.0 - float(np.abs(grid - np.fliplr(grid)).mean() / denom)
    vertical = 1.0 - float(np.abs(grid - np.flipud(grid)).mean() / denom)
    rot180 = 1.0 - float(np.abs(grid - np.flipud(np.fliplr(grid))).mean() / denom)

    return {
        "feat_symmetry_horizontal": horizontal,
        "feat_symmetry_vertical": vertical,
        "feat_symmetry_rot180": rot180,
    }


def _frequency_energy(grid: np.ndarray) -> Dict[str, float]:
    spectrum = np.fft.fft2(grid)
    power = np.abs(spectrum) ** 2
    total_energy = float(power.sum())
    if total_energy == 0:
        return {"feat_fft_energy_total_log": 0.0, "feat_fft_high_ratio": 0.0}

    shifted = np.fft.fftshift(power)
    rows, cols = grid.shape
    r_idx, c_idx = np.indices((rows, cols))
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    distances = np.sqrt((r_idx - center_r) ** 2 + (c_idx - center_c) ** 2)
    cutoff = min(rows, cols) * 0.25
    high_energy = float(shifted[distances >= cutoff].sum())
    low_energy = float(shifted[distances < cutoff].sum())

    return {
        "feat_fft_energy_total_log": float(np.log1p(total_energy)),
        "feat_fft_high_ratio": float(high_energy / total_energy),
        "feat_fft_low_ratio": float(low_energy / total_energy),
    }


def extract_features(grid: np.ndarray, hist_bins: int = DEFAULT_HIST_BINS) -> Dict[str, float]:
    """Compute a fixed-length feature vector for an NS8 grid."""
    features: Dict[str, float] = OrderedDict()
    features.update(_normalized_histogram(grid, bins=hist_bins))
    features.update(_row_col_stats(grid))
    features.update(_axis_autocorr(grid, axis=0, lags=(1, 2)))
    features.update(_axis_autocorr(grid, axis=1, lags=(1, 2)))
    features.update(_symmetry_scores(grid))
    features.update(_frequency_energy(grid))
    return features


__all__ = ["extract_features", "DEFAULT_HIST_BINS"]
