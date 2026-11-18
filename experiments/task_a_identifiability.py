"""One fixed Task A exact-feature identifiability audit; no model fits."""
from collections import Counter
from fractions import Fraction
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from ns8lab.features import extract_features
from ns8lab.grids import generate_grid, VALID_VIEWS

NS = (16, 24, 32)
KS = tuple(range(1, 9))
VIEWS = ("TLF", "TRF", "BLF", "BRF", "TLB", "BLB", "BRB", "TRB")


def schema():
    return ([f"feat_hist_bin_{i:02d}" for i in range(16)] +
            ["feat_hist_entropy", "feat_hist_gini"] +
            ["feat_" + name for name in ["row_mean_mean", "row_mean_std", "col_mean_mean", "col_mean_std",
             "row_var_mean", "row_var_std", "col_var_mean", "col_var_std", "row_skew_mean", "row_skew_std",
             "col_skew_mean", "col_skew_std", "row_kurt_mean", "row_kurt_std", "col_kurt_mean", "col_kurt_std",
             "value_min", "value_max", "value_mean", "value_std"]] +
            [f"feat_autocorr_axis{axis}_lag{lag}" for axis in [0, 1] for lag in [1, 2]] +
            ["feat_symmetry_horizontal", "feat_symmetry_vertical", "feat_symmetry_rot180",
             "feat_fft_energy_total_log", "feat_fft_high_ratio", "feat_fft_low_ratio"])


def exact_key(row):
    values = tuple(float(x) for x in row)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("nonempty finite feature input required")
    return tuple(x.hex() for x in values)  # Signed zero retained, no rounding.


def group_inputs(keys, labels, order):
    if not keys or len(keys) != len(labels) or any(label not in order for label in labels):
        raise ValueError("nonempty aligned inputs and declared labels required")
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    result = []
    for members in groups.values():
        counts = Counter(labels[i] for i in members)
        maximum = max(counts.values())
        result.append({"group": len(result), "representative_row": members[0], "members": members,
                       "label_counts": {label: counts[label] for label in order},
                       "maximum_correct": maximum, "maximizing_labels": [label for label in order if counts[label] == maximum],
                       "conflicting_labels": len(counts) > 1})
    maximum = sum(g["maximum_correct"] for g in result)
    ratio = Fraction(maximum, len(labels))
    return {"groups": result, "maximum_correct": maximum, "rows": len(labels),
            "ceiling_fraction": {"numerator": ratio.numerator, "denominator": ratio.denominator},
            "ceiling_decimal": float(ratio)}


def approximate_pairs(X, labels):
    # Pairwise diagnostic only: no union-find or inferred hard equivalence.
    pairs = []
    for i, j in itertools.combinations(range(len(X)), 2):
        if np.all(np.isclose(X[i], X[j], atol=1e-12, rtol=0)):
            pairs.append({"rows": [i, j], "labels_differ": labels[i] != labels[j],
                          "bit_exact": exact_key(X[i]) == exact_key(X[j]),
                          "max_absolute_difference": float(np.max(np.abs(X[i] - X[j])))})
    return pairs


def protocol():
    return {"task": "task_a_view", "N": list(NS), "k": list(KS), "views": list(VIEWS),
            "rows": 192, "raw_cells": 118784, "budget": {"max_rows": 192, "max_raw_cells": 200000, "max_pairs": 20000},
            "hist_bins": 16, "input": "canonical extract_features only,48source-ordered binary64 columns; metadata/raw arrays NOT X",
            "exact_equality": "tuple(float(value).hex()) across ALL48dimensions; signed zero retained; no hash/rounding/tolerance grouping",
            "raw_control_equality": "same shape and exact little-endian int64 row-major logical values, independent of memory layout",
            "extractor_layout": "use actual native generator arrays unchanged; record strides/C/F flags, do not coerce contiguous layout",
            "approximate": {"atol": 1e-12, "rtol": 0, "pairwise_only": True, "no_ceiling_no_components": True},
            "ceiling": "whole-table deterministic oracle maximum=sum_g max_label_count(g)/rows; all ties/members retained, not fitted/heldout/population accuracy",
            "raw_feature_relation": "do not assume partitions nest; report raw aliases split by bit-exact features, plus exact-X collisions between distinct raw values",
            "stopping": "source/schema/nonfinite/budget mismatch fails; one run only; no fit/search/representation revision",
            "historical": "TaskA .59/.44 remain prior reports with distinct domain/provenance limits; no relabeling as broken"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_pins(source):
    names = ["src/ns8lab/features.py", "src/ns8lab/grids.py", "experiments/task_a_identifiability.py",
             "tests/test_task_a_identifiability.py", "docs/TASK_A_IDENTIFIABILITY.md", "pyproject.toml"]
    return {name: digest((source / name).read_bytes()) for name in names}


def freeze(source, out):
    if tuple(VALID_VIEWS) != VIEWS or len(schema()) != 48:
        raise ValueError("canonical views/schema mismatch")
    if len(NS) * len(KS) * len(VIEWS) != 192 or sum(n*n for n in NS)*len(KS)*len(VIEWS) > 200000:
        raise ValueError("budget mismatch")
    out.mkdir(exist_ok=False)
    save(out / "PROTOCOL.json", protocol()); save(out / "SCHEMA.json", schema())
    save(out / "SOURCE-PINS.json", source_pins(source))
    save(out / "RUNTIME.json", {"host": platform.node(), "python": sys.version, "numpy": np.__version__,
         "source_origins": {"generate_grid": sys.modules[generate_grid.__module__].__file__, "extract_features": sys.modules[extract_features.__module__].__file__}})
    files = {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}
    save(out / "PRE-RUN-FREEZE.json", {"files": files, "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "feature_rows_computed": 0, "actual_table_collisions_computed": False, "fits": 0})
    return {"status": "FROZEN_BEFORE_FEATURE_OUTCOMES", "rows_budget": 192, "raw_cells": 118784}


def run(source, out):
    frozen = json.loads((out / "PRE-RUN-FREEZE.json").read_text())
    def unchanged():
        for name, sha in frozen["files"].items():
            if digest((out / name).read_bytes()) != sha:
                raise ValueError("frozen input changed")
        if source_pins(source) != json.loads((out / "SOURCE-PINS.json").read_text()):
            raise ValueError("frozen source changed")
    unchanged()
    if (out / "RESULT.json").exists():
        raise ValueError("previous result exists; no rerun")
    save(out / "STARTED.json", {"started_unix": time.time(), "freeze_sha256": digest((out / "PRE-RUN-FREEZE.json").read_bytes())})
    rows, features, raw_keys, raw_controls = [], [], [], []
    start = time.perf_counter()
    for n, k, view in itertools.product(NS, KS, VIEWS):
        grid = generate_grid(n, k, view)
        values = extract_features(grid, hist_bins=16)
        if list(values) != schema():
            raise ValueError("feature schema changed")
        x = list(values.values()); exact_key(x)
        raw = grid.astype("<i8").tobytes(order="C")
        i = len(rows)
        rows.append({"row": i, "N": n, "k": k, "view": view, "target": view})
        features.append(x); raw_keys.append((n, raw))
        raw_controls.append({"row": i, "shape": [n, n], "values": grid.ravel(order="C").tolist(),
                             "rowmajor_sha256": digest(raw), "strides": list(grid.strides),
                             "C_contiguous": bool(grid.flags.c_contiguous), "F_contiguous": bool(grid.flags.f_contiguous)})
    extraction_seconds = time.perf_counter() - start
    if len(rows) != 192 or sum(r["N"]**2 for r in rows) != 118784:
        raise ValueError("actual work budget mismatch")
    keys = [exact_key(x) for x in features]; labels = [r["target"] for r in rows]
    exact = group_inputs(keys, labels, VIEWS); raw = group_inputs(raw_keys, labels, VIEWS)
    feature_group = {i: group["group"] for group in exact["groups"] for i in group["members"]}
    raw_group = {i: group["group"] for group in raw["groups"] for i in group["members"]}
    raw_splits = [{"raw_group": group["group"], "members": group["members"], "exact_feature_groups": sorted({feature_group[i] for i in group["members"]})}
                  for group in raw["groups"] if len({feature_group[i] for i in group["members"]}) > 1]
    intrinsic, additional = [], []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if labels[i] != labels[j] and raw_keys[i] == raw_keys[j]:
            intrinsic.append({"rows": [i, j], "raw_group": raw_group[i], "exact_features_equal": keys[i] == keys[j]})
        if labels[i] != labels[j] and keys[i] == keys[j] and raw_keys[i] != raw_keys[j]:
            additional.append({"rows": [i, j], "feature_group": feature_group[i], "raw_groups": [raw_group[i], raw_group[j]]})
    approx = approximate_pairs(np.array(features), labels)
    save(out / "ROWS.json", rows); save(out / "FEATURES.json", {"names": schema(), "X": features, "hex": keys})
    save(out / "RAW-CONTROLS.json", raw_controls)
    save(out / "EXACT-FEATURE-GROUPS.json", exact); save(out / "RAW-GROUPS.json", raw)
    save(out / "WITNESSES.json", {"intrinsic_raw_label_alias_pairs": intrinsic, "distinct_raw_exact_X_label_conflicts": additional, "logical_raw_groups_split_by_exact_features": raw_splits})
    save(out / "APPROXIMATE-PAIRS.json", {"policy": protocol()["approximate"], "pairs": approx, "ceiling": None})
    unchanged()
    result = {"status": "COMPLETE_ONE_NO_FIT_AUDIT", "rows": len(rows), "raw_cells": 118784, "feature_dimensions": 48, "fits": 0,
              "exact_feature_groups": len(exact["groups"]), "conflicting_exact_groups": sum(g["conflicting_labels"] for g in exact["groups"]),
              "exact_empirical_ceiling": {k: v for k, v in exact.items() if k != "groups"},
              "raw_empirical_ceiling": {k: v for k, v in raw.items() if k != "groups"},
              "intrinsic_raw_label_alias_pairs": len(intrinsic), "distinct_raw_exact_X_label_conflict_pairs": len(additional),
              "logical_raw_groups_split_by_exact_features": len(raw_splits), "pairwise_atol_matches_DIAGNOSTIC_ONLY": len(approx),
              "extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - start,
              "freeze_sha256": digest((out / "PRE-RUN-FREEZE.json").read_bytes()),
              "limits": "Whole-table bit-exact empirical oracle bound, not fitted/heldout/population accuracy. Approximate pairs yield no hard ceiling; layout/roundoff may change exact groups. TaskA origin aliases are design limits, not a broken task."}
    save(out / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze(source, args.out) if args.mode == "freeze" else run(source, args.out), indent=2))
