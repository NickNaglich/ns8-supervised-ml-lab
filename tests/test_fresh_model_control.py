"""Fake features/models exercise the harness without fits or real extraction."""
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from experiments import fresh_model_control as fresh


@pytest.fixture
def fake_preparation(monkeypatch, tmp_path):
    out = tmp_path / "prepared"
    calls = dict(native=0, canonical=0, generate=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("a real model/scaler fit or model construction was attempted")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(fresh, "model", forbidden)
    monkeypatch.setattr(fresh, "source_pins", lambda: {"src/ns8lab/features.py": "fake-source"})
    monkeypatch.setattr(fresh, "peak_bytes", lambda: 12345)

    def grid(n, k, view):
        assert (out / "PREPARATION-FREEZE.json").exists()
        assert (out / "SPLIT.json").exists()
        calls["generate"] += 1
        return np.full((n, n), k, dtype=np.int64)

    def features(g, hist_bins, variant):
        assert hist_bins == 16
        calls[variant] += 1
        return {name: float(i + int(g[0, 0])) for i, name in enumerate(fresh.schema())}

    monkeypatch.setattr(fresh, "generate_grid", grid)
    monkeypatch.setattr(fresh, "extract_features", lambda g, hist_bins: features(g, hist_bins, "native"))
    monkeypatch.setattr(fresh, "c_order_features", lambda g, hist_bins: features(g, hist_bins, "canonical"))
    return out, calls


def test_prepare_validate_no_fits_and_frozen_split(fake_preparation):
    out, calls = fake_preparation
    result = fresh.prepare(out)
    assert result["counts"] == dict(native_extractions=192, canonical_extractions=192, classifier_fits=0, scaler_fits=0)
    assert calls == dict(native=192, canonical=192, generate=192)
    before = dict(calls)
    assert fresh.validate(out)["fits"] == 0
    assert calls == before  # Validation does not re-extract.
    split = fresh.load(out / "SPLIT.json")
    assert len(split["train"]["rows"]) == 144 and len(split["test"]["rows"]) == 48
    assert not set(split["train"]["groups"]) & set(split["test"]["groups"])
    assert set(split["train"]["label_counts"]) == set(split["test"]["label_counts"]) == set(fresh.VIEWS)
    assert fresh.load(out / "RAW-ALIASES.json")  # Retained, never relabeled/deduplicated.
    assert all(n.startswith("feat_") for n in fresh.load(out / "FEATURES.json")["names"])
    assert fresh.load(out / "IDENTITIES.json")["canonical"]["feature_variant"] == "c_order_int64_v1"


def test_existing_prepare_output_refuses_before_work(fake_preparation):
    out, calls = fake_preparation
    out.mkdir()
    with pytest.raises(FileExistsError):
        fresh.prepare(out)
    assert calls == dict(native=0, canonical=0, generate=0)
    assert list(out.iterdir()) == []


def test_extraction_failure_preserves_freeze_and_counts(fake_preparation, monkeypatch):
    out, _ = fake_preparation
    monkeypatch.setattr(fresh, "extract_features", lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic failure")))
    with pytest.raises(ValueError, match="synthetic failure"):
        fresh.prepare(out)
    assert (out / "PREPARATION-FREEZE.json").exists()
    assert fresh.load(out / "FAILURE.json")["counts"]["native_extractions"] == 1
    assert not (out / "PREPARED.json").exists()


def test_split_missing_class_rejected():
    records = fresh.rows()
    for r in records:
        if r["target"] == fresh.VIEWS[-1]:
            r["target"] = fresh.VIEWS[0]
    with pytest.raises(ValueError, match="class support"):
        fresh.split_rows(records)


def test_raw_origin_overlap_and_hash_rejected():
    records = fresh.rows(); split = fresh.split_rows(records)
    raw = []
    for r in records:
        a = np.ones(r["N"]**2, dtype="<i8")
        raw.append(dict(row=r["row"], values=a.tolist(), rowmajor_sha256=fresh.sha(a.tobytes())))
    with pytest.raises(ValueError, match="raw origin crosses"):
        fresh.raw_evidence(records, raw, split)
    raw[0]["rowmajor_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        fresh.raw_evidence(records, raw, split)


@pytest.mark.parametrize("kind", ["bytes", "split", "schema", "nan", "short"])
def test_corruption_fails_before_fitting(fake_preparation, kind):
    out, _ = fake_preparation; fresh.prepare(out)
    name = "SPLIT.json" if kind == "split" else "FEATURES.json"
    d = fresh.load(out / name)
    if kind == "bytes":
        (out / name).write_text("{}")  # No integrity re-pin: first guard rejects.
    else:
        if kind == "split": d["test"]["rows"][0] = d["train"]["rows"][0]
        if kind == "schema": d["names"][0] = "feature_pipeline_id"
        if kind == "nan": d["native_hex"][0][0] = "nan"
        if kind == "short": d["canonical_hex"][0].pop()
        (out / name).write_text(json.dumps(d))
        manifest = fresh.load(out / "PREPARED.json")
        manifest["files"][name] = fresh.sha((out / name).read_bytes())
        # For split, deliberately bypass both integrity layers to reach semantic guard.
        if kind == "split":
            frozen = fresh.load(out / "PREPARATION-FREEZE.json")
            frozen[name] = manifest["files"][name]
            (out / "PREPARATION-FREEZE.json").write_text(json.dumps(frozen))
            manifest["files"]["PREPARATION-FREEZE.json"] = fresh.sha((out / "PREPARATION-FREEZE.json").read_bytes())
        (out / "PREPARED.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fresh.fit(out, out.parent / "fit")
    assert not (out.parent / "fit").exists()


@pytest.mark.parametrize("kind", ["source", "runtime"])
def test_source_or_runtime_drift_requires_new_run(fake_preparation, monkeypatch, kind):
    out, _ = fake_preparation; fresh.prepare(out)
    monkeypatch.setattr(fresh, "source_pins" if kind == "source" else "runtime", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source/runtime drift"):
        fresh.fit(out, out.parent / "fit")


def test_explicit_fit_fake_models_train_only_shared_indices(fake_preparation, monkeypatch):
    out, calls = fake_preparation; fresh.prepare(out)
    split = fresh.load(out / "SPLIT.json"); trained = []; scaled = []
    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
    from sklearn.pipeline import make_pipeline
    class FakeScaler(TransformerMixin, BaseEstimator):
        def fit(self, X, y):
            scaled.append(X.copy()); self.mean_ = X.mean(axis=0)
            assert X.shape == (144, 48) and len(y) == 144
            return self
        def transform(self, X):
            return X - self.mean_
    class FakeClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            trained.append((X.copy(), y.copy())); self.classes_ = np.unique(y)
            return self
        def predict(self, X):
            assert X.shape == (48, 48)
            return np.array([fresh.VIEWS[0]] * len(X))
    monkeypatch.setattr(fresh, "model", lambda: make_pipeline(FakeScaler(), FakeClassifier()))
    before = dict(calls); result = fresh.fit(out, out.parent / "fit")
    assert calls == before and len(trained) == 2
    assert np.array_equal(trained[0][0], trained[1][0])
    data = fresh.load(out / "FEATURES.json")
    original = np.array([[float.fromhex(v) for v in row] for row in data["native_hex"]])
    assert len(scaled) == 2 and np.array_equal(scaled[0], original[split["train"]["rows"]])
    assert np.array_equal(trained[0][0], scaled[0] - scaled[0].mean(axis=0))
    expected_y = np.array([fresh.rows()[i]["target"] for i in split["train"]["rows"]])
    assert np.array_equal(trained[0][1], expected_y) and np.array_equal(trained[1][1], expected_y)
    assert result["counts"] == dict(classifier_fits_started=2, scaler_fits_started=2)
    with pytest.raises(FileExistsError):
        fresh.fit(out, out.parent / "fit")
    assert len(trained) == 2


def test_real_model_configuration_constructs_but_does_not_fit():
    pipeline = fresh.model()
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    assert isinstance(scaler, StandardScaler) and isinstance(classifier, LogisticRegression)
    assert all(classifier.get_params()[k] == v for k, v in fresh.PARAMS.items())
    assert not hasattr(scaler, "mean_") and not hasattr(classifier, "coef_")


@pytest.mark.parametrize("name", ["FEATURES.json", "SPLIT.json", "IDENTITIES.json", "PREPARED.json"])
def test_change_after_validation_cannot_reach_estimator(fake_preparation, monkeypatch, name):
    out, _ = fake_preparation; fresh.prepare(out)
    original_validate = fresh.validate
    constructions = []

    def validate_then_change(path):
        result = original_validate(path)
        target = path / name
        data = fresh.load(target)
        if name == "FEATURES.json": data["native_hex"][0][0] = float(12345).hex()
        if name == "SPLIT.json": data["test"]["rows"][0] = data["train"]["rows"][0]
        if name == "IDENTITIES.json": data["native"]["variant"] = "unchecked"
        if name == "PREPARED.json": data["seconds"] = -1
        target.write_text(json.dumps(data))
        return result

    def forbidden_model():
        constructions.append(True)
        pytest.fail("an estimator was constructed after post-validation input drift")

    monkeypatch.setattr(fresh, "validate", validate_then_change)
    monkeypatch.setattr(fresh, "model", forbidden_model)
    target = out.parent / "fit"
    with pytest.raises(ValueError, match="changed"):
        fresh.fit(out, target)
    assert constructions == [] and not target.exists()


def test_partial_fit_failure_not_relabelled_complete(fake_preparation, monkeypatch):
    out, _ = fake_preparation; fresh.prepare(out)
    class FakeModel:
        calls = 0
        def fit(self, X, y):
            FakeModel.calls += 1
            if FakeModel.calls == 2: raise RuntimeError("second fake fit failed")
        def predict(self, X): return np.array([fresh.VIEWS[0]] * len(X))
    monkeypatch.setattr(fresh, "model", FakeModel)
    target = out.parent / "fit"
    with pytest.raises(RuntimeError, match="second fake fit"):
        fresh.fit(out, target)
    assert (target / "native-RESULT.json").exists() and not (target / "COMPLETE.json").exists()
    assert fresh.load(target / "FAILURE.json")["counts"]["classifier_fits_started"] == 2


def test_cli_requires_explicit_command_and_routes_no_fit(monkeypatch, tmp_path, capsys):
    commands = []
    monkeypatch.setattr(fresh, "prepare", lambda p: commands.append(("prepare", p)) or {"fits": 0})
    monkeypatch.setattr(fresh, "validate", lambda p: commands.append(("validate", p)) or {"fits": 0})
    monkeypatch.setattr(fresh, "fit", lambda *a: pytest.fail("automatic fit"))
    with pytest.raises(SystemExit): fresh.main([])
    fresh.main(["prepare", "--out", str(tmp_path / "new")])
    fresh.main(["validate", "--out", str(tmp_path / "new")])
    assert [x[0] for x in commands] == ["prepare", "validate"]
    assert '"fits": 0' in capsys.readouterr().out
