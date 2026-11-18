"""One controlled storage-layout intervention, then conditional saved-table trial."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from ns8lab.features import extract_features as native_extract
from experiments.task_a_identifiability import exact_key, group_inputs, schema, VIEWS

VARIANT = "experimental_c_order_v1_20261117"
INPUT_HASHES = {
    "ROWS.json": "7e6f0864ca66e2fb8974c33dca014a9c764b22d2f2fa85ca518b8b3c359171a0",
    "FEATURES.json": "8735e6b5d2f9ac329a4617fb3a4c268757aa2e342a921285ed300e8563fdba17",
    "RAW-CONTROLS.json": "e7701f08239f4386c95bb656f25362114f72965eca927dc7fedb221309f94e15",
    "RAW-GROUPS.json": "4674c910c6d1d4f3dfb2dc721f05da9fc9cf98792a077b95adaee9c6002e97c5",
    "RESULT.json": "57a63681380b0aa6374ee58e8385cfe421161973be81fd7f79306d1a1ffa6208",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def signature(grid):
    return {"shape": list(grid.shape), "dtype": grid.dtype.str,
            "rowmajor_sha256": digest(grid.tobytes(order="C"))}


def controls(base):
    if type(base) is not np.ndarray or base.ndim != 2 or base.shape[0] != base.shape[1] or base.dtype != np.dtype("int64"):
        raise ValueError("square int64 array required")
    backing = np.array(base[::-1], copy=True, order="C")
    return {"C": np.array(base, copy=True, order="C"),
            "F": np.array(base, copy=True, order="F"), "negative_stride": backing[::-1]}


def c_order_features(grid):
    """Opt-in storage copy only; formulas/schema/hist_bins/dtype unchanged."""
    if type(grid) is not np.ndarray or grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.dtype != np.dtype("int64"):
        raise ValueError("square int64 array required; no dtype coercion")
    return native_extract(np.array(grid, copy=True, order="C"), hist_bins=16)


def source_pins(source):
    names = ["src/ns8lab/features.py", "src/ns8lab/data.py", "pyproject.toml",
             "experiments/task_a_identifiability.py", "experiments/layout_stability.py",
             "tests/test_layout_stability.py", "docs/LAYOUT_STABILITY.md"]
    return {name: digest((source / name).read_bytes()) for name in names}


def inputs_unchanged(saved):
    for name, sha in INPUT_HASHES.items():
        if digest((saved / name).read_bytes()) != sha:
            raise ValueError("accepted saved input drift: " + name)


def freeze(source, saved, out):
    inputs_unchanged(saved)
    raw = json.loads((saved / "RAW-CONTROLS.json").read_text())
    old = json.loads((saved / "FEATURES.json").read_text())
    rows = json.loads((saved / "ROWS.json").read_text())
    if len(raw) != 192 or len(rows) != 192 or old["names"] != schema() or len(old["X"]) != 192:
        raise ValueError("saved table/schema mismatch")
    if raw[66]["values"] != raw[71]["values"] or raw[66]["shape"] != [24, 24]:
        raise ValueError("fixed logical control mismatch")
    if sum(len(g["values"]) for g in raw) != 118784:
        raise ValueError("saved cell budget mismatch")
    out.mkdir(exist_ok=False)
    pins = source_pins(source)
    save(out / "PROTOCOL.json", {
        "feature_variant": VARIANT, "native_formula_sha256": pins["src/ns8lab/features.py"],
        "wrapper_source_sha256": pins["experiments/layout_stability.py"],
        "native_identity": "unversioned feature schema; project0.0.1; CSV/parquet has no feature-variant token",
        "controls": "saved row66/71 N24k1; equal int64 values, C/F/reversed-backing-negative-stride",
        "phase1_gate": "nativeC andF match SAVEDrow66/71 respectively and differ; else STOP before wrapper",
        "wrapper": "ONE np.array(copy=True,order=C) trial; no dtype/formula/label/default changes",
        "success": "all48float.hex identical across3wrapper controls; values/dtype/nonmutation verified, NOT accuracy improvement",
        "phase2": "ONLY if justified:192saved matrices,118784cells, wrapper once each; compare SAVEDoldX, reuse exact group logic",
        "budget": {"direct_native": 3, "wrapper_controls": 3, "wrapper_table": 192, "total_extractions": 198, "fits": 0},
        "equality": "all48 ordered float.hex including signed zero; no rounding/atol merging",
        "native_defaults_and_old_models_artifacts_caches": "untouched; separate experimental output identity required",
        "limits": "controlled same-values intervention on ONE matrix; exact groups runtime-bound; old19/24 only oldX; intrinsic view aliases remain; no learned/population claim"})
    save(out / "SOURCE-PINS.json", pins); save(out / "INPUT-PINS.json", INPUT_HASHES)
    save(out / "SCHEMA.json", schema())
    save(out / "RUNTIME.json", {"host": platform.node(), "python": sys.version, "numpy": np.__version__,
         "native_origin": sys.modules[native_extract.__module__].__file__})
    files = {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}
    save(out / "PRE-OUTCOME-FREEZE.json", {"files": files, "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "actual_extractions": 0, "outcomes_computed": False})
    return {"status": "FROZEN_BEFORE_INTERVENTION", "variant": VARIANT, "input_hashes": INPUT_HASHES}


def run(source, saved, out):
    frozen = json.loads((out / "PRE-OUTCOME-FREEZE.json").read_text())
    def unchanged():
        inputs_unchanged(saved)
        for name, sha in frozen["files"].items():
            if digest((out / name).read_bytes()) != sha:
                raise ValueError("frozen trial file changed")
        if source_pins(source) != json.loads((out / "SOURCE-PINS.json").read_text()):
            raise ValueError("frozen source changed")
    unchanged()
    if (out / "RESULT.json").exists():
        raise ValueError("prior trial outcome exists")
    save(out / "STARTED.json", {"time_unix": time.time(), "freeze_sha256": digest((out / "PRE-OUTCOME-FREEZE.json").read_bytes())})
    start = time.perf_counter()
    raw = json.loads((saved / "RAW-CONTROLS.json").read_text())
    old = json.loads((saved / "FEATURES.json").read_text())
    rows = json.loads((saved / "ROWS.json").read_text())
    base = np.array(raw[66]["values"], dtype=np.int64).reshape(24, 24)
    arrays = controls(base)
    counts = {"direct_native": 0, "wrapper_controls": 0, "wrapper_table": 0}
    def extract(grid, wrapped):
        before = signature(grid)
        tick = time.perf_counter()
        values = c_order_features(grid) if wrapped else native_extract(grid, hist_bins=16)
        seconds = time.perf_counter() - tick
        if signature(grid) != before or list(values) != schema():
            raise ValueError("mutation/dtype/value/schema mismatch")
        return {"features": list(values.values()), "hex": exact_key(values.values()),
                "input": before, "strides": list(grid.strides), "C": bool(grid.flags.c_contiguous),
                "F": bool(grid.flags.f_contiguous), "seconds": seconds}
    for grid in arrays.values():
        if signature(grid) != signature(base):
            raise ValueError("same-value control failure")
    def finish(decision, **extra):
        unchanged()
        result = {"decision": decision, "feature_variant": VARIANT, "counts": counts,
                  "total_extractions": sum(counts.values()), "fits": 0,
                  "elapsed_seconds": time.perf_counter() - start, "freeze_sha256": digest((out / "PRE-OUTCOME-FREEZE.json").read_bytes()), **extra}
        save(out / "RESULT.json", result)
        return result
    native = {}
    for name, grid in arrays.items():
        native[name] = extract(grid, False); counts["direct_native"] += 1
    save(out / "NATIVE-CONTROLS.json", native)
    reproduced = (list(native["C"]["hex"]) == old["hex"][66] and
                  list(native["F"]["hex"]) == old["hex"][71] and native["C"]["hex"] != native["F"]["hex"])
    if not reproduced:
        return finish("STOP_native_difference_not_reproduced", reproduction=False)
    wrapped = {}
    for name, grid in arrays.items():
        wrapped[name] = extract(grid, True); counts["wrapper_controls"] += 1
    save(out / "WRAPPER-CONTROLS.json", {"feature_variant": VARIANT, "layouts": wrapped})
    if len({record["hex"] for record in wrapped.values()}) != 1:
        return finish("STOP_wrapper_stability_failed", reproduction=True)
    features, keys, changes, costs = [], [], [], []
    for i, record in enumerate(raw):
        if record["row"] != i or record["shape"] != [rows[i]["N"], rows[i]["N"]] or any(type(v) is not int or not 1 <= v <= rows[i]["N"] for v in record["values"]):
            raise ValueError("saved row/value mismatch")
        grid = np.array(record["values"], dtype=np.int64).reshape(record["shape"])
        value = extract(grid, True); counts["wrapper_table"] += 1
        features.append(value["features"]); keys.append(value["hex"]); costs.append(value["seconds"])
        different = [c for c in range(48) if value["hex"][c] != old["hex"][i][c]]
        if different:
            changes.append({"row": i, "columns": [schema()[c] for c in different],
                            "old_hex": [old["hex"][i][c] for c in different], "new_hex": [value["hex"][c] for c in different]})
    assert counts == {"direct_native": 3, "wrapper_controls": 3, "wrapper_table": 192}
    groups = group_inputs(keys, [r["target"] for r in rows], VIEWS)
    raw_groups = json.loads((saved / "RAW-GROUPS.json").read_text())
    raw_splits = [g["group"] for g in raw_groups["groups"] if len({keys[i] for i in g["members"]}) > 1]
    save(out / "VARIANT-FEATURES.json", {"feature_variant": VARIANT, "names": schema(), "X": features, "hex": keys})
    save(out / "VARIANT-GROUPS.json", {"feature_variant": VARIANT, **groups})
    save(out / "CHANGED-ROWS.json", changes); save(out / "EXTRACTION-COSTS.json", costs)
    return finish("RETAIN_OPT_IN_TRIAL" if not raw_splits else "STOP_raw_alias_instability", reproduction=True,
                  wrapper_controls_stable=True, changed_rows=len(changes), logical_raw_groups_still_split=raw_splits,
                  old_empirical_ceiling=json.loads((saved / "RESULT.json").read_text())["exact_empirical_ceiling"],
                  variant_empirical_ceiling={k: v for k, v in groups.items() if k != "groups"},
                  variant_exact_groups=len(groups["groups"]), table_extraction_seconds=sum(costs),
                  limits="Retain experimental storage variant only, not default adoption. Whole-table oracle bound, no fit/heldout/population benefit; intrinsic source-view aliases remain.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "run"])
    parser.add_argument("--saved-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); source = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze(source, args.saved_audit, args.out) if args.mode == "freeze" else run(source, args.saved_audit, args.out), indent=2))
