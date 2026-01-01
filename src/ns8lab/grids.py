"""Canonical Naglich Squares (NS8) grid generation.

Deterministic, vectorized generation of A-family NS8 grids parameterized by
matrix size (N), kernel slope (k), and an orientation/view label. Values are
1-indexed and live in the range [1, N].
"""

from __future__ import annotations

from typing import Dict

import numpy as np

VALID_VIEWS = (
    "TLF",
    "TRF",
    "BLF",
    "BRF",
    "TLB",
    "BLB",
    "BRB",
    "TRB",
)
_VIEW_SET = set(VALID_VIEWS)


def _validate_inputs(N: int, k: int, view: str) -> str:
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    normalized_view = view.upper()
    if normalized_view not in _VIEW_SET:
        raise ValueError(f"view must be one of {list(VALID_VIEWS)}")
    return normalized_view


def _base_kernel(N: int, k: int) -> np.ndarray:
    """Base A_TLF kernel: (row + k * col) mod N, 1-indexed to [1, N]."""
    rows, cols = np.indices((N, N))
    kernel = (rows + k * cols) % N
    return kernel + 1


def _apply_view_transform(grid: np.ndarray, view: str) -> np.ndarray:
    """Apply orientation transforms from the canonical TLF view.

    Encoding:
    - First letter (T/B): vertical orientation; B flips vertically.
    - Second letter (L/R): horizontal orientation; R flips horizontally.
    - Third letter (F/B): depth; B applies transpose.
    """
    oriented = grid
    if view[0] == "B":  # Top/Bottom switch: vertical flip
        oriented = np.flipud(oriented)
    if view[1] == "R":  # Left/Right switch: horizontal flip
        oriented = np.fliplr(oriented)
    if view[2] == "B":  # Front/Back switch: transpose
        oriented = oriented.T
    return oriented


def generate_grid(N: int, k: int, view: str = "TLF") -> np.ndarray:
    """Generate an NS8 grid for size N, kernel slope k, and orientation view."""
    normalized_view = _validate_inputs(N, k, view)
    base = _base_kernel(N, k)
    return _apply_view_transform(base, normalized_view).astype(np.int64)


def generate_all_views(N: int, k: int) -> Dict[str, np.ndarray]:
    """Return a dict of all 8 dihedral orientations for the given (N, k)."""
    base = _base_kernel(N, k)
    return {view: _apply_view_transform(base, view) for view in VALID_VIEWS}


def a_tlf(N: int, k: int) -> np.ndarray:
    """Canonical A-family grid in the top-left-front (TLF) orientation."""
    return generate_grid(N, k, view="TLF")


def a_trb(N: int, k: int) -> np.ndarray:
    """Canonical A-family grid in the top-right-back (TRB) orientation."""
    return generate_grid(N, k, view="TRB")
