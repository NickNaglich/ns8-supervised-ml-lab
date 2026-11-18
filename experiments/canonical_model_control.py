"""One frozen native-versus-C-order classifier control: exactly two fits."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ns8lab.feature_variants import c_order_features, pipeline_metadata
from ns8lab.validation import build_groups, feature_columns
from experiments.task_a_identifiability import exact_key, VIEWS, schema

PARAMS = {"C": 1., "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4, "random_state": 0}
HASHES = {"ROWS.json": "7e6f0864ca66e2fb8974c33dca014a9c764b22d2f2fa85ca518b8b3c359171a0",
          "FEATURES.json": "8735e6b5d2f9ac329a4617fb3a4c268757aa2e342a921285ed300e8563fdba17",
          "RAW-CONTROLS.json": "e7701f08239f4386c95bb656f25362114f72965eca927dc7fedb221309f94e15",
          "RAW-GROUPS.json": "4674c910c6d1d4f3dfb2dc721f05da9fc9cf98792a077b95adaee9c6002e97c5"}
PRIOR_C_HASH = "fc4af5bc69c014034fbd6a35f6feaf98d4283784eb258d3cbf29580be7ef3591"
MAX_BYTES = 16 * 1024**2
MEMORY_TARGET = 256 * 1024**2


def sha(raw): return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def peak_memory():
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [(n, ctypes.c_size_t) for n in ["peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]
        m = PMC(); m.cb = ctypes.sizeof(m)
        k = ctypes.WinDLL("kernel32"); k.GetCurrentProcess.restype = ctypes.c_void_p
        p = ctypes.WinDLL("psapi"); p.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
        if not p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(m), ctypes.sizeof(m)):
            raise RuntimeError("own-process memory measurement unavailable")
        return int(m.peak_ws)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def frame(rows, values, names):
    df = pd.DataFrame(rows)
    for i, name in enumerate(names): df[name] = [row[i] for row in values]
    return df


def matrix(df, names):
    if feature_columns(df) != names: raise ValueError("ordered feature schema mismatch")
    return np.array(df[names].to_numpy(dtype=float), dtype=np.float64, order="C", copy=True)


def raw_overlap(raw_groups, train, test):
    tr, te = set(train), set(test)
    return [g["group"] for g in raw_groups["groups"] if tr.intersection(g["members"]) and te.intersection(g["members"])]


def qualification(values, expected):
    if len(values) != len(expected): return {"match": False, "rows": "length mismatch"}
    different = [{"row": i, "columns": [c for c, (a, b) in enumerate(zip(exact_key(row), expected[i])) if a != b]}
                 for i, row in enumerate(values) if len(row) != len(expected[i]) or list(exact_key(row)) != expected[i]]
    return {"match": not different, "different": different, "rows_checked": len(values), "dimensions": 48}


def fit_once(model, X, y, train, test):
    model.fit(X[train], y[train])
    return model.predict(X[test]), model.decision_function(X[test])


def pins(source):
    names = ["src/ns8lab/features.py", "src/ns8lab/feature_variants.py", "src/ns8lab/validation.py",
             "experiments/task_a_identifiability.py", "experiments/canonical_model_control.py",
             "tests/test_canonical_model_control.py", "docs/CANONICAL_MODEL_CONTROL.md"]
    return {n: sha((source/n).read_bytes()) for n in names}


def input_guard(audit, prior):
    for name, expected in HASHES.items():
        if sha((audit/name).read_bytes()) != expected: raise ValueError("saved audit drift: " + name)
    if sha(prior.read_bytes()) != PRIOR_C_HASH: raise ValueError("prior C-order table drift")


def freeze(source, audit, prior, out):
    input_guard(audit, prior)
    out.mkdir(exist_ok=False)
    rows = json.loads((audit/"ROWS.json").read_text()); old = json.loads((audit/"FEATURES.json").read_text())
    if len(rows) != 192 or old["names"] != schema() or any(r["row"] != i for i, r in enumerate(rows)):
        raise ValueError("cohort/schema mismatch")
    df = frame(rows, old["X"], old["names"]); X = matrix(df, old["names"])
    if [list(exact_key(x)) for x in X] != old["hex"]: raise ValueError("saved native hex mismatch")
    y = df.target.to_numpy(); groups = build_groups(df, "nk")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0).split(X, y, groups))
    overlaps = raw_overlap(json.loads((audit/"RAW-GROUPS.json").read_text()), tr, te)
    admitted = (len(set(groups)) == 24 and len(set(groups[tr])) == 18 and len(set(groups[te])) == 6 and
                len(tr) == 144 and len(te) == 48 and not set(groups[tr]) & set(groups[te]) and
                set(y[tr]) == set(y[te]) == set(VIEWS) and not overlaps)
    def members(indices):
        return {"rows": indices.tolist(), "groups": groups[indices].tolist(),
                "label_counts": {v: int(sum(y[indices] == v)) for v in VIEWS}}
    save(out/"SPLIT.json", {"train": members(tr), "test": members(te), "raw_overlap_groups": overlaps, "admitted": admitted})
    save(out/"SCHEMA.json", old["names"]); save(out/"SOURCE-PINS.json", pins(source))
    save(out/"INPUT-PINS.json", {"audit": HASHES, "prior_C192": PRIOR_C_HASH})
    save(out/"RUNTIME.json", {"host": platform.node(), "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "peak_bytes": peak_memory()})
    save(out/"CANONICAL-IDENTITY.json", pipeline_metadata(16))
    save(out/"PROTOCOL.json", {"cohort": "192saved TaskA rows,N16/24/32,k1..8,all8views; native48X reused", "group": "N,k moduloN, seed0,test_size.25; shared fixed144/48 indices, no CV", "model_params": PARAMS,
        "full_estimator_defaults": LogisticRegression(**PARAMS).get_params(), "scaler": StandardScaler().get_params(),
        "X": "exact48source-order float64 columns; shared C-order tabular matrix; metadata outside X",
        "qualification_gate": "192current canonical extractions, all48hex equal priorC192 BEFORE fits; mismatch STOP",
        "fits": {"classifier": 2, "preprocessor": 2}, "memory_target_bytes": MEMORY_TARGET, "output_limit_bytes": MAX_BYTES, "process_timeout_seconds": 120,
        "metrics": "accuracy,macroF1 labels all8; per-row predictions/exact max-decision ties/confusions/warnings/time; no expected improvement/resampling/retry",
        "identity": "legacy saved native provenance versus CURRENT C-order pipeline metadata, bundled separately with source/data/split/params; no old identity relabel",
        "limits": "This heldout partition only, no significance/population/learned advantage claim. Preserve all source-origin aliases and negatives."})
    files = {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir())}
    save(out/"PRE-TRIAL-FREEZE.json", {"files": files, "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "extractions": 0, "classifier_fits": 0, "outcomes_computed": False})
    if not admitted: raise ValueError("split differs from fixed admission; STOP before extraction/fits")
    return {"status": "FROZEN", "train_groups": sorted(set(groups[tr])), "test_groups": sorted(set(groups[te])), "rows": [len(tr), len(te)]}


def run(source, audit, prior, out):
    f = json.loads((out/"PRE-TRIAL-FREEZE.json").read_text())
    def unchanged():
        input_guard(audit, prior)
        if pins(source) != json.loads((out/"SOURCE-PINS.json").read_text()): raise ValueError("source drift")
        for name, expected in f["files"].items():
            if sha((out/name).read_bytes()) != expected: raise ValueError("frozen protocol drift")
    unchanged()
    save(out/"STARTED.json", {"started_unix": time.time(), "freeze_sha256": sha((out/"PRE-TRIAL-FREEZE.json").read_bytes())})
    counts = {"canonical_extractions_started": 0, "canonical_extractions_completed": 0, "classifier_fits": 0, "preprocessor_fits": 0}
    start = time.perf_counter(); events = (out/"EVENTS.jsonl").open("x", encoding="utf-8")
    def event(stage):
        events.write(json.dumps({"stage": stage, "counts": counts.copy(), "seconds": time.perf_counter()-start})+"\n"); events.flush()
    def limits():
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if size > MAX_BYTES: raise RuntimeError("output budget exceeded; retain partials")
        if peak_memory() > MEMORY_TARGET: raise RuntimeError("memory target exceeded; retain partials")
    def finish(status, **extra):
        unchanged(); limits(); event(status)
        result = {"status": status, "counts": counts, "elapsed_seconds": time.perf_counter()-start, "peak_memory_bytes": peak_memory(), "fits": counts["classifier_fits"], "freeze_sha256": sha((out/"PRE-TRIAL-FREEZE.json").read_bytes()), **extra}
        save(out/"RESULT.json", result); events.close(); return result
    limits()
    rows = json.loads((audit/"ROWS.json").read_text()); raw = json.loads((audit/"RAW-CONTROLS.json").read_text()); old = json.loads((audit/"FEATURES.json").read_text())
    split = json.loads((out/"SPLIT.json").read_text()); tr = np.array(split["train"]["rows"]); te = np.array(split["test"]["rows"])
    values = []; before = time.perf_counter()
    with (out/"CANONICAL-PARTIAL.jsonl").open("x", encoding="utf-8") as partial:
        for i, record in enumerate(raw):
            grid = np.array(record["values"], dtype=np.int64).reshape(record["shape"])
            original = grid.tobytes(order="C"); counts["canonical_extractions_started"] += 1; event("extract_start")
            features = c_order_features(grid, 16)
            if list(features) != old["names"] or grid.tobytes(order="C") != original: raise ValueError("schema/value drift")
            x = list(features.values()); exact_key(x); values.append(x); counts["canonical_extractions_completed"] += 1
            partial.write(json.dumps({"row": i, "features": x, "hex": exact_key(x)})+"\n"); partial.flush(); event("extract_complete")
    extraction_seconds = time.perf_counter()-before
    check = qualification(values, json.loads(prior.read_text())["hex"]); save(out/"QUALIFICATION.json", check)
    if not check["match"]: return finish("STOP_BEFORE_FITS_feature_mismatch")
    metadata = pipeline_metadata(16)
    if metadata != json.loads((out/"CANONICAL-IDENTITY.json").read_text()): raise ValueError("pipeline identity drift")
    save(out/"CANONICAL-FEATURES.json", {"metadata": metadata, "names": old["names"], "X": values, "hex": [exact_key(x) for x in values]})
    matrices = {"native_saved": matrix(frame(rows, old["X"], old["names"]), old["names"]), "c_order_int64_v1": matrix(frame(rows, values, old["names"]).assign(**metadata), old["names"])}
    y = np.array([r["target"] for r in rows]); results = []; predictions = []
    native_identity = {"kind": "saved_native_legacy_unversioned", "features_artifact_sha256": HASHES["FEATURES.json"], "producer_commit": "252dce226bf17a979f8fbbdfba9719a66aed6e85", "note": "Saved original X, no new pipeline identity asserted"}
    original_lr = LogisticRegression.fit; original_scaler = StandardScaler.fit
    def classifier_fit(self, *args, **kwargs):
        counts["classifier_fits"] += 1; event("classifier_fit")
        if counts["classifier_fits"] > 2: raise RuntimeError("classifier fit budget exceeded")
        return original_lr(self, *args, **kwargs)
    def scaler_fit(self, *args, **kwargs):
        counts["preprocessor_fits"] += 1; event("preprocessor_fit")
        if counts["preprocessor_fits"] > 2: raise RuntimeError("preprocessor fit budget exceeded")
        return original_scaler(self, *args, **kwargs)
    LogisticRegression.fit = classifier_fit; StandardScaler.fit = scaler_fit
    try:
        for name, X in matrices.items():
            limits(); model = make_pipeline(StandardScaler(), LogisticRegression(**PARAMS)); tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); pred, scores = fit_once(model, X, y, tr, te)
            seconds = time.perf_counter()-tick; scaler = model.named_steps["standardscaler"]; classifier = model.named_steps["logisticregression"]
            if int(scaler.n_samples_seen_) != len(tr): raise AssertionError("scaler training support mismatch")
            identity = native_identity if name == "native_saved" else metadata
            result = {"variant": name, "accuracy": float(accuracy_score(y[te], pred)), "macro_f1": float(f1_score(y[te], pred, labels=list(VIEWS), average="macro", zero_division=0)),
                "labels": list(VIEWS), "confusion": confusion_matrix(y[te], pred, labels=list(VIEWS)).tolist(), "classifier_classes": classifier.classes_.tolist(), "iterations": classifier.n_iter_.tolist(),
                "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught], "fit_predict_seconds": seconds,
                "scaler_train_rows": len(tr), "scaler_mean": scaler.mean_.tolist(), "feature_identity": identity}
            results.append(result)
            for position, row in enumerate(te):
                decisions = scores[position]; ties = classifier.classes_[decisions == max(decisions)].tolist()
                predictions.append({"variant": name, "row": int(row), "N": rows[row]["N"], "k": rows[row]["k"], "target": str(y[row]), "prediction": str(pred[position]), "decision_scores": decisions.tolist(), "exact_max_decision_labels": ties})
            save(out/(name+"-RESULT.json"), result)
            bundle_metadata = {"feature_identity": identity, "schema": old["names"], "source_pins": pins(source), "source_commit": f["source_commit"], "input_pins": {"audit": HASHES, "prior_C192": PRIOR_C_HASH}, "split": split, "model_params": PARAMS, "freeze_sha256": sha((out/"PRE-TRIAL-FREEZE.json").read_bytes())}
            with (out/(name+".joblib")).open("xb") as stream: joblib.dump({"metadata": bundle_metadata, "model": model}, stream)
            event("model_saved"); limits()
    finally:
        LogisticRegression.fit = original_lr; StandardScaler.fit = original_scaler
    assert counts == {"canonical_extractions_started": 192, "canonical_extractions_completed": 192, "classifier_fits": 2, "preprocessor_fits": 2}
    save(out/"PREDICTIONS.json", predictions)
    return finish("COMPLETE_ONE_CONTROL", results=results, extraction_seconds=extraction_seconds,
                  delta_canonical_minus_native={key: results[1][key]-results[0][key] for key in ["accuracy", "macro_f1"]},
                  limits="One frozen Nk heldout partition only; no CV/search/retry/significance/population claim. Source-view aliases retained; no native192 replay or old identity relabel.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("mode", choices=["freeze", "run"])
    p.add_argument("--audit", type=Path, required=True); p.add_argument("--prior-corder", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); source = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze(source, a.audit, a.prior_corder, a.out) if a.mode == "freeze" else run(source, a.audit, a.prior_corder, a.out), indent=2))
