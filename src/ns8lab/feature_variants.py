"""Explicit opt-in storage variant; native formulas/defaults remain unchanged."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from . import features as native_module

C_ORDER_VARIANT = "c_order_int64_v1"
FEATURE_VARIANTS = ("native", C_ORDER_VARIANT)
IDENTITY_COLUMNS = ("feature_variant", "feature_pipeline_id", "feature_pipeline_spec")


def c_order_features(grid, hist_bins):
    if type(grid) is not np.ndarray or grid.ndim != 2 or not grid.shape[0] or grid.shape[0] != grid.shape[1] or grid.dtype != np.dtype("int64"):
        raise ValueError("C-order variant requires a nonempty square int64 array; no dtype coercion")
    return native_module.extract_features(np.array(grid, copy=True, order="C"), hist_bins=hist_bins)


def pipeline_metadata(hist_bins):
    if type(hist_bins) is not int or hist_bins < 1:
        raise ValueError("C-order variant requires positive integer hist_bins")
    spec = {
        "schema_version": 1, "feature_variant": C_ORDER_VARIANT, "hist_bins": hist_bins,
        "native_source_sha256": hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest(),
        "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "numpy": np.__version__, "platform": platform.platform(), "byteorder": sys.byteorder},
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return dict(zip(IDENTITY_COLUMNS, [C_ORDER_VARIANT, hashlib.sha256(encoded.encode("ascii")).hexdigest(), encoded]))
