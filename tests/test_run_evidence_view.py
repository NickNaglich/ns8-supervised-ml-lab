"""Saved-JSON reader controls; no grid/feature extraction, model, or Streamlit."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_evidence_utils", ROOT / "ui/utils.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def partition(rows, mode="n"):
    return {"row_positions": rows, "sample_ids": [f"row-{r}" for r in rows],
            "groups": None if mode == "none" else [r // 2 for r in rows]}


def split_record(mode="n"):
    return {"schema_version": 1, "group_mode": mode, "group_definition": mode,
            "feature_columns": ["feat_value_max", "feat_shape"],
            "outer": {"train": partition([0, 1, 2, 3], mode), "test": partition([4, 5], mode)},
            "inner_splitter": "GroupKFold" if mode != "none" else "StratifiedKFold",
            "inner": [{"train": partition([0, 1], mode), "validation": partition([2, 3], mode)},
                      {"train": partition([2, 3], mode), "validation": partition([0, 1], mode)}]}


def write_run(path, *, signature=None, split=True, baseline=False, task="task_a_view"):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"task": task, "group_mode": "n"}))
    (path / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))
    if signature is not None:
        (path / "dataset_signature.txt").write_text(signature)
    if split:
        (path / "split.json").write_text(json.dumps(split_record()))
    if baseline:
        (path / "baseline.json").write_text(json.dumps({"name": "saved_identity", "prediction": [4, 4],
            "metrics": {"rmse": 0.0, "r2": 1.0}, "scope": "test record, not a performance run"}))
    return path


def test_hand_counted_split_baseline_and_read_only(tmp_path):
    path = write_run(tmp_path / "run", baseline=True, task="task_c_regress_n")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    details = u.load_run_details(str(path))
    evidence = details["evidence"]
    summary = evidence["split_summary"]
    assert summary["train"] == {"rows": 4, "groups": 2}
    assert summary["test"] == {"rows": 2, "groups": 1}
    assert summary["feature_count"] == 2
    assert len(summary["inner_folds"]) == 2
    assert summary["inner_folds"][0] == {"train": {"rows": 2, "groups": 1}, "validation": {"rows": 2, "groups": 1}}
    assert evidence["baseline"]["value"]["metrics"]["rmse"] == 0
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_matching_ids_and_config_without_data_identity_is_unverified(tmp_path):
    a, b = [u.load_run_evidence(write_run(tmp_path / n)) for n in ("a", "b")]
    result = u.compare_run_evidence(a, b)
    assert result["status"] == "unverified"
    assert next(f for f in result["facts"] if f["dimension"] == "recorded_dataset_identity")["status"] == "unverified"


def test_match_is_only_recorded_evidence_not_authentication(tmp_path):
    a, b = [u.load_run_evidence(write_run(tmp_path / n, signature="same-recorded-file-hash")) for n in ("a", "b")]
    result = u.compare_run_evidence(a, b)
    assert result["status"] == "matched_recorded_evidence"
    assert "not authenticated" in result["scope"]
    assert a["baseline"]["status"] == "missing"  # native classification omission is legitimate


@pytest.mark.parametrize("dimension", ["task", "data", "schema", "test", "train", "group"])
def test_mismatch_dominates_unknown(tmp_path, dimension):
    a, b = [u.load_run_evidence(write_run(tmp_path / n, signature="same")) for n in ("a", "b")]
    if dimension == "task":
        b["task"] = "task_b_kbucket"
        a["dataset_identity"]["status"] = "missing"
    elif dimension == "data":
        b["dataset_identity"]["value"] = "different"
    elif dimension == "schema":
        b["split"]["value"]["feature_columns"].reverse()
    elif dimension in ("test", "train"):
        b["split"]["value"]["outer"][dimension]["sample_ids"][0] = "different"
    else:
        b["split"]["value"]["group_mode"] = "nk"
    assert u.compare_run_evidence(a, b)["status"] == "incompatible"


def test_legacy_missing_split_and_baseline_is_unknown(tmp_path):
    e = u.load_run_evidence(write_run(tmp_path / "old", split=False))
    assert e["split"]["status"] == e["baseline"]["status"] == "missing"
    assert u.compare_run_evidence(e, e)["status"] == "unverified"


@pytest.mark.parametrize("bad", ['[]', '{broken', '{"metrics":{"x":NaN}}'])
def test_malformed_records_are_invalid_not_a_match(tmp_path, bad):
    path = write_run(tmp_path / "run")
    (path / "split.json").write_text(bad)
    e = u.load_run_evidence(path)
    assert e["split"]["status"] == "invalid"
    assert u.compare_run_evidence(e, e)["status"] == "unverified"


@pytest.mark.parametrize("kind", ["outer_overlap", "inner_overlap", "group_overlap", "duplicate_row", "wrong_inner_id"])
def test_invalid_memberships_are_not_counted_as_valid(tmp_path, kind):
    path = write_run(tmp_path / "run")
    s = split_record()
    if kind == "outer_overlap":
        s["outer"]["test"] = partition([0, 5])
    elif kind == "inner_overlap":
        s["inner"][0]["validation"] = partition([0, 3])
    elif kind == "group_overlap":
        s["outer"]["test"]["groups"][0] = 0
    elif kind == "duplicate_row":
        s["outer"]["train"]["row_positions"][1] = 0
    else:
        s["inner"][0]["train"]["sample_ids"][0] = "invented"
    (path / "split.json").write_text(json.dumps(s))
    e = u.load_run_evidence(path)
    assert e["split"]["status"] == "invalid"
    assert e["split_summary"] is None


def test_row_split_null_groups_and_reordered_membership(tmp_path):
    path = write_run(tmp_path / "run")
    (path / "split.json").write_text(json.dumps(split_record("none")))
    e = u.load_run_evidence(path)
    assert e["split_summary"]["train"]["groups"] is None
    other = copy.deepcopy(e)
    outer = other["split"]["value"]["outer"]["test"]
    outer["row_positions"].reverse(); outer["sample_ids"].reverse()
    assert all(f["status"] != "mismatch" for f in u.compare_run_evidence(e, other)["facts"])


def test_conflicting_signature_and_invalid_baseline(tmp_path):
    path = write_run(tmp_path / "run", signature="sidecar", baseline=True)
    (path / "metrics.json").write_text(json.dumps({"dataset_signature": "different"}))
    (path / "baseline.json").write_text(json.dumps({"name": "bad", "prediction": [1], "metrics": {"rmse": 0}}))
    e = u.load_run_evidence(path)
    assert e["dataset_identity"]["status"] == "conflicting"
    assert e["baseline"]["status"] == "invalid"
    assert u.compare_run_evidence(e, e)["status"] == "unverified"


def test_large_evidence_is_not_read(tmp_path, monkeypatch):
    path = write_run(tmp_path / "run")
    (path / "split.json").write_text(" " * (u._EVIDENCE_LIMIT + 1))
    original = Path.read_text
    def read(p, *a, **kw):
        assert p.name != "split.json"
        return original(p, *a, **kw)
    monkeypatch.setattr(Path, "read_text", read)
    assert u.load_run_evidence(path)["split"]["status"] == "invalid"


class FakeStreamlit:
    def __init__(self):
        self.calls = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def columns(self, count):
        return [self] * count
    def expander(self, *a, **kw):
        return self
    def selectbox(self, label, options, **kw):
        return options[1] if kw.get("key") == "compare_b" else options[0]
    def text_input(self, *a, **kw):
        return ""
    def checkbox(self, *a, **kw):
        return False
    def metric(self, *a, **kw):
        pytest.fail("gain-style delta must not render")
    def __getattr__(self, name):
        def record(*args, **kw):
            self.calls.append((name, args))
        return record


def test_actual_ui_details_and_comparison_with_fake_surface(tmp_path, monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    package = types.ModuleType("ui"); package.__path__ = []
    monkeypatch.setitem(sys.modules, "ui", package)
    monkeypatch.setitem(sys.modules, "ui.utils", u)
    spec = importlib.util.spec_from_file_location("evidence_app", ROOT / "ui/app.py")
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    a = write_run(tmp_path / "a", baseline=True, task="task_c_regress_n")
    b = write_run(tmp_path / "b", task="task_a_view")
    index = [{"run_id": "a", "run_path": str(a), "task": "task_c_regress_n"},
             {"run_id": "b", "run_path": str(b), "task": "task_a_view"}]
    app.section_run_details(index)
    assert any(name == "json" and isinstance(args[0], dict) and args[0].get("prediction_count") == 2 for name, args in fake.calls)
    app.section_compare_runs(index)
    assert any("incompatible" in str(args) for name, args in fake.calls)
    assert any(name == "dataframe" and isinstance(args[0], list) and args[0] and args[0][0].get("metric") == "accuracy" for name, args in fake.calls)


def test_reader_does_not_import_training_or_model_loaders():
    # Other full-suite tests legitimately import models during collection.
    # Inspect the actual reader in a fresh process, not pytest's shared modules.
    import subprocess
    reader = (ROOT / "ui/utils.py").resolve()
    code = r"""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

class NoModelImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sklearn", "joblib", "ns8lab"}:
            self.attempted.append(fullname)
            raise AssertionError("reader attempted forbidden import: " + fullname)
        return None

guard = NoModelImports()
sys.meta_path.insert(0, guard)
selected = Path(sys.argv[1]).resolve(strict=True)
spec = importlib.util.spec_from_file_location("isolated_run_evidence", selected)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert Path(module.__file__).resolve() == selected
assert Path(module.__spec__.origin).resolve() == selected
assert Path(module.load_run_evidence.__code__.co_filename).resolve() == selected
assert not guard.attempted, guard.attempted
loaded = [name for name in sys.modules if name.split(".")[0] in {"sklearn", "joblib", "ns8lab"}]
assert not loaded, loaded
print(json.dumps({"reader": str(selected), "forbidden_imports": guard.attempted, "loaded_model_modules": loaded}))
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(reader)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report == {"reader": str(reader), "forbidden_imports": [], "loaded_model_modules": []}


@pytest.mark.parametrize("raw", ['[]', '{broken', '{"accuracy":NaN}'])
def test_invalid_metric_objects_do_not_break_details_or_signature_reader(tmp_path, raw):
    path = write_run(tmp_path / "run")
    (path / "metrics.json").write_text(raw)
    details = u.load_run_details(str(path))
    assert details["metrics"] == {}
    assert u.load_dataset_signature_from_run(path) is None
    assert u.compare_run_evidence(details["evidence"], details["evidence"])["status"] == "unverified"


def test_boolean_schema_version_is_not_native_schema_one(tmp_path):
    path = write_run(tmp_path / "run")
    record = split_record(); record["schema_version"] = True
    (path / "split.json").write_text(json.dumps(record))
    assert u.load_run_evidence(path)["split"]["status"] == "invalid"
