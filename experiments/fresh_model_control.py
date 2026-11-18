"""Fresh synthetic TaskA control. Preparation/validation never fit a model."""
import argparse
from collections import Counter
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from ns8lab.features import extract_features
from ns8lab.feature_variants import c_order_features, pipeline_metadata
from ns8lab.grids import generate_grid, VALID_VIEWS
from ns8lab.validation import build_groups, feature_columns
from experiments.task_a_identifiability import schema, exact_key

NS = (16, 24, 32)
KS = tuple(range(1, 9))
VIEWS = tuple(VALID_VIEWS)
PARAMS = dict(C=1., solver="lbfgs", max_iter=1000, tol=1e-4, random_state=0)
MAX_BYTES = 16 * 1024**2
SOURCE_FILES = ("experiments/fresh_model_control.py", "experiments/task_a_identifiability.py",
                "src/ns8lab/grids.py", "src/ns8lab/features.py",
                "src/ns8lab/feature_variants.py", "src/ns8lab/validation.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_pins():
    root = Path(__file__).resolve().parents[1]
    return {name: sha((root / name).read_bytes()) for name in SOURCE_FILES}


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__,
                platform=platform.platform(), byteorder=sys.byteorder)


def peak_bytes():
    if sys.platform != "win32":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
            (n, ctypes.c_size_t) for n in ("peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    m = PMC(); m.cb = ctypes.sizeof(m)
    kernel = ctypes.WinDLL("kernel32"); kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(m), ctypes.sizeof(m)):
        raise RuntimeError("own-process memory measurement unavailable")
    return int(m.peak_ws)


def rows():
    return [dict(row=i, N=n, k=k, view=v, target=v)
            for i, (n, k, v) in enumerate(itertools.product(NS, KS, VIEWS))]


def split_rows(records):
    df = pd.DataFrame(records)
    groups = build_groups(df, "nk")
    y = df.target.to_numpy()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0).split(
        np.zeros((len(df), 1)), y, groups))
    if set(groups[train]) & set(groups[test]) or set(y[train]) != set(VIEWS) or set(y[test]) != set(VIEWS):
        raise ValueError("group separation/class support failed")
    if (len(df), len(set(groups)), len(set(groups[train])), len(set(groups[test])), len(train), len(test)) != (192, 24, 18, 6, 144, 48):
        raise ValueError("fixed cohort/split counts changed; no resampling")
    return {part: dict(rows=idx.tolist(), groups=groups[idx].tolist(),
                       label_counts=dict(Counter(y[idx]))) for part, idx in (("train", train), ("test", test))}


def raw_evidence(records, raw, split):
    groups = {}
    if len(raw) != len(records):
        raise ValueError("raw row count mismatch")
    for i, (r, item) in enumerate(zip(records, raw)):
        n = r["N"]
        vals = item["values"]
        if len(vals) != n*n or any(type(x) is not int or not 1 <= x <= n for x in vals):
            raise ValueError("invalid saved logical raw grid")
        key = np.array(vals, dtype="<i8").tobytes()
        if item["row"] != i or sha(key) != item["rowmajor_sha256"]:
            raise ValueError("raw row/hash mismatch")
        groups.setdefault((n, key), []).append(i)
    tr, te = set(split["train"]["rows"]), set(split["test"]["rows"])
    if any(tr.intersection(m) and te.intersection(m) for m in groups.values()):
        raise ValueError("identical logical raw origin crosses train/test")
    return [dict(members=m, labels=dict(Counter(records[i]["target"] for i in m)))
            for m in groups.values() if len(m) > 1]


def protocol():
    return dict(version="fresh_task_a_control_v1", N=list(NS), k=list(KS), views=list(VIEWS),
                rows=192, raw_cells=118784, hist_bins=16, feature_names=schema(),
                split="GroupShuffleSplit(N,k modulo N), seed0,test_size0.25; fixed shared indices",
                models=dict(params=PARAMS, scaler="StandardScaler fit on training rows only"),
                interfaces="48 engineered feat_* only; raw arrays, labels, groups and identity are NOT X",
                budget=dict(native_extractions=192, canonical_extractions=192, prepare_fits=0,
                            explicit_fit_classifiers=2, explicit_fit_scalers=2, output_bytes=MAX_BYTES,
                            process_seconds_target=120),
                provenance="fresh generated data, NOT historical exact replay or an old pipeline identity",
                limits="One synthetic grouped partition; raw view-origin aliases retained. No expected improvement, tuning, significance or population claim.")


def files_size(out):
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if size > MAX_BYTES:
        raise ValueError("output exceeds 16MiB budget")
    return size


def prepare(out):
    out = Path(out)
    out.mkdir(exist_ok=False)  # Refuse existing output before generation/extraction.
    started = time.perf_counter()
    counts = dict(native_extractions=0, canonical_extractions=0, classifier_fits=0, scaler_fits=0)
    try:
        records = rows(); split = split_rows(records)
        save(out / "PROTOCOL.json", protocol())
        save(out / "SOURCE.json", source_pins()); save(out / "RUNTIME.json", runtime())
        save(out / "ROWS.json", records); save(out / "SPLIT.json", split)
        save(out / "IDENTITIES.json", {"native": {"variant": "fresh_native", "source_sha256": source_pins()["src/ns8lab/features.py"], "runtime": runtime(), "hist_bins": 16},
                                        "canonical": pipeline_metadata(16)})
        save(out / "PREPARATION-FREEZE.json", {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir())})
        native, canonical, raw = [], [], []
        for r in records:
            grid = generate_grid(r["N"], r["k"], view=r["view"])
            if grid.dtype != np.dtype("int64") or grid.shape != (r["N"], r["N"]):
                raise ValueError("generator dtype/shape changed")
            before = grid.tobytes(order="C")
            counts["native_extractions"] += 1; a = extract_features(grid, hist_bins=16)
            counts["canonical_extractions"] += 1; b = c_order_features(grid, hist_bins=16)
            if list(a) != schema() or list(b) != schema() or before != grid.tobytes(order="C"):
                raise ValueError("feature schema/input mutation")
            native.append(exact_key(a.values())); canonical.append(exact_key(b.values()))
            raw.append(dict(row=r["row"], values=grid.ravel(order="C").tolist(),
                            rowmajor_sha256=sha(grid.astype("<i8", copy=False).tobytes(order="C")),
                            strides=list(grid.strides), C=bool(grid.flags.c_contiguous), F=bool(grid.flags.f_contiguous)))
        aliases = raw_evidence(records, raw, split)
        save(out / "RAW.json", raw)
        save(out / "FEATURES.json", dict(names=schema(), native_hex=native, canonical_hex=canonical))
        save(out / "RAW-ALIASES.json", aliases)
        if load(out / "SOURCE.json") != source_pins():
            raise ValueError("source changed during preparation")
        save(out / "PREPARED.json", dict(status="PREPARED_NO_FIT", counts=counts,
             seconds=time.perf_counter()-started, peak_process_bytes=peak_bytes(),
             files={p.name: sha(p.read_bytes()) for p in sorted(out.iterdir())}))
        return dict(status="PREPARED_NO_FIT", counts=counts, output_bytes=files_size(out))
    except Exception as exc:
        save(out / "FAILURE.json", dict(error=type(exc).__name__, message=str(exc), counts=counts))
        raise


def validate(out):
    out = Path(out)
    files_size(out)
    prepared = load(out / "PREPARED.json")
    if prepared["status"] != "PREPARED_NO_FIT" or (out / "FAILURE.json").exists():
        raise ValueError("not a complete prepared dataset")
    for name, pin in prepared["files"].items():
        if Path(name).name != name or sha((out / name).read_bytes()) != pin:
            raise ValueError("prepared data/protocol drift: " + name)
    if load(out / "SOURCE.json") != source_pins() or load(out / "RUNTIME.json") != runtime():
        raise ValueError("source/runtime drift; prepare a separately identified run")
    if load(out / "PROTOCOL.json") != protocol() or load(out / "ROWS.json") != rows():
        raise ValueError("fixed protocol/cohort changed")
    for name, pin in load(out / "PREPARATION-FREEZE.json").items():
        if Path(name).name != name or sha((out / name).read_bytes()) != pin:
            raise ValueError("pre-extraction freeze mismatch")
    records = rows(); split = load(out / "SPLIT.json")
    if split != split_rows(records):
        raise ValueError("fixed split changed; no resampling")
    data = load(out / "FEATURES.json")
    if data["names"] != schema() or feature_columns(pd.DataFrame(columns=data["names"])) != schema():
        raise ValueError("feature-only ordered schema mismatch")
    for key in ("native_hex", "canonical_hex"):
        if len(data[key]) != 192 or any(len(x) != 48 or list(exact_key(float.fromhex(v) for v in x)) != x for x in data[key]):
            raise ValueError("malformed/nonfinite/noncanonical feature hex")
    if raw_evidence(records, load(out / "RAW.json"), split) != load(out / "RAW-ALIASES.json"):
        raise ValueError("raw alias evidence changed")
    return dict(status="VALID_NO_FIT", rows=192, features=48, train_rows=144, test_rows=48, fits=0)


def model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(**PARAMS))


def fit(prepared, out):
    prepared, out = Path(prepared), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("fit output already exists")
    manifest_bytes = (prepared / "PREPARED.json").read_bytes()
    validate(prepared)  # No private inputs or extraction; refuse drift before fitting.
    if (prepared / "PREPARED.json").read_bytes() != manifest_bytes:
        raise ValueError("prepared manifest changed during validation")
    manifest = json.loads(manifest_bytes)
    snapshot = {}
    for name in ("SPLIT.json", "FEATURES.json", "IDENTITIES.json"):
        raw = (prepared / name).read_bytes()
        if sha(raw) != manifest["files"][name]:
            raise ValueError("validated fit input changed: " + name)
        snapshot[name] = json.loads(raw)
    out.mkdir(exist_ok=False)
    split, data = snapshot["SPLIT.json"], snapshot["FEATURES.json"]
    train, test = split["train"]["rows"], split["test"]["rows"]
    y = np.array([r["target"] for r in rows()])
    save(out / "FIT-FREEZE.json", dict(prepared_sha256=sha(manifest_bytes),
         files=manifest["files"], identities=snapshot["IDENTITIES.json"],
         source=source_pins(), runtime=runtime(), params=PARAMS, split=split,
         limits=protocol()["limits"]))
    counts = dict(classifier_fits_started=0, scaler_fits_started=0)
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        for variant in ("native", "canonical"):
            X = np.array([[float.fromhex(v) for v in row] for row in data[variant+"_hex"]], dtype=float)
            estimator = model(); start = time.perf_counter()
            counts["classifier_fits_started"] += 1; counts["scaler_fits_started"] += 1
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(X[train], y[train]); prediction = estimator.predict(X[test])
            save(out / (variant+"-RESULT.json"), dict(variant=variant, test_rows=test, true=y[test].tolist(),
                 prediction=prediction.tolist(), labels=list(VIEWS), accuracy=float(accuracy_score(y[test], prediction)),
                 macroF1=float(f1_score(y[test], prediction, labels=list(VIEWS), average="macro", zero_division=0)),
                 confusion=confusion_matrix(y[test], prediction, labels=list(VIEWS)).tolist(),
                 warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                 seconds=time.perf_counter()-start))
        validate(prepared)
        save(out / "COMPLETE.json", dict(status="TWO_EXPLICIT_FITS", counts=counts, peak_process_bytes=peak_bytes()))
        return dict(status="TWO_EXPLICIT_FITS", counts=counts, output_bytes=files_size(out))
    except Exception as exc:
        save(out / "FAILURE.json", dict(error=type(exc).__name__, message=str(exc), counts=counts))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "fit"):
        sub = subs.add_parser(name)
        sub.add_argument("--out", required=True, type=Path, help="new output directory, or prepared directory for validate")
        if name == "fit":
            sub.add_argument("--prepared", required=True, type=Path)
    args = parser.parse_args(argv)
    result = fit(args.prepared, args.out) if args.command == "fit" else {"prepare": prepare, "validate": validate}[args.command](args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
