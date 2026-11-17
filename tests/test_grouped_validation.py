import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ns8lab import train, tune
from ns8lab.validation import arithmetic_baseline, build_groups, prepare_split


def frame(task="task_a_view"):
    records = []
    for n in (4, 8, 12, 16):
        for k in (1, 2):
            for label in (0, 1):
                records.append(dict(sample_id=len(records), N=n, k=k, view=str(label),
                                    target=float(n) if task == "task_c_regress_n" else label,
                                    feat_value_max=float(n), feat_shape=float(k + label)))
    return pd.DataFrame(records)


def test_nested_groups_and_original_memberships():
    df = frame()
    split = prepare_split(df, "task_a_view", 0.25, 0, "n")
    outer = split.provenance["outer"]
    assert set(outer["train"]["groups"]).isdisjoint(outer["test"]["groups"])
    assert set(outer["train"]["row_positions"]) | set(outer["test"]["row_positions"]) == set(range(len(df)))
    for fold in split.provenance["inner"]:
        assert set(fold["train"]["groups"]).isdisjoint(fold["validation"]["groups"])
        assert set(fold["train"]["row_positions"]) | set(fold["validation"]["row_positions"]) == set(split.train)
    assert split.provenance["feature_columns"] == ["feat_value_max", "feat_shape"]
    assert set(build_groups(pd.DataFrame({"N": [4, 4], "k": [1, 5]}), "nk")) == {"4_1"}


@pytest.mark.parametrize("mode", ["none", "n", "nk"])
def test_declared_mode_is_preserved(mode):
    split = prepare_split(frame(), "task_a_view", 0.25, 0, mode, 2)
    assert split.provenance["group_mode"] == mode
    assert (split.train_groups is None) == (mode == "none")
    assert split.provenance["inner_splitter"] == ("StratifiedKFold" if mode == "none" else "GroupKFold")


@pytest.mark.parametrize("kind,message", [("one_group", "outer split"), ("one_training_group", "inner CV"),
                                         ("heldout_class", "every class"), ("inner_class", "inner CV")])
def test_infeasible_contracts_fail(kind, message):
    df = frame()
    if kind == "one_group":
        df["N"] = 32
    elif kind == "one_training_group":
        df = df[df.N.isin([4, 8])]
    elif kind == "heldout_class":
        df["target"] = df.N
    else:
        df["target"] = ((df.N.isin([12, 16])) & (df.view == "1")).astype(int)
    with pytest.raises(ValueError, match=message):
        prepare_split(df, "task_b_kbucket", 0.25, 0, "n")


@pytest.mark.parametrize("module", [train, tune])
@pytest.mark.parametrize("task", ["task_a_view", "task_c_regress_n"])
def test_real_entrypoints_feed_cv_groups_and_train_only_preprocessing(module, task, tmp_path, monkeypatch):
    df = frame(task)
    monkeypatch.setattr(module, "build_dataset", lambda **kwargs: df.copy())
    for name in ("RUNS_DIR", "RESULTS_DIR", "FIGURES_DIR", "EXPERIMENTS_DIR"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, tmp_path / name)
    classifier = task == "task_a_view"
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5) if classifier else Ridge())
    if module is train:
        monkeypatch.setattr(module, "_make_classifiers" if classifier else "_make_regressors",
                            lambda seed: {"small": (pipe, {})})
    else:
        monkeypatch.setattr(module, "_model_registry", lambda seed, task: {"small": (pipe, "unused")})
    seen = []
    class CheckedSearch(GridSearchCV):
        def fit(self, X, y=None, **params):
            groups = params["groups"]
            assert groups is not None
            for tr, va in self.cv.split(X, y, groups):
                assert set(groups[tr]).isdisjoint(groups[va])
            seen.append(groups.tolist())
            # Real sklearn search/pipeline, bounded to one process for tests.
            self.n_jobs = 1
            return super().fit(X, y, **params)
    monkeypatch.setattr(module, "GridSearchCV", CheckedSearch)
    scaled_inputs = []
    real_scaler_fit = StandardScaler.fit
    def track_scaler_fit(self, X, y=None, **kwargs):
        scaled_inputs.append(np.array(X))
        return real_scaler_fit(self, X, y, **kwargs)
    monkeypatch.setattr(StandardScaler, "fit", track_scaler_fit)
    if module is train:
        summaries = module.train_and_evaluate(task, seed=0, group_mode="n", test_size=0.25)
    else:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump(dict(task=task, seed=0, group_mode="n", test_size=0.25,
                                           models=[dict(name="small", param_grid={})])))
        summaries = module.tune_from_config(cfg)
    assert len(seen) == len(summaries) == 1
    run = module.RUNS_DIR / summaries[0]["run_id"]
    membership = json.loads((run / "split.json").read_text())
    assert seen[0] == membership["outer"]["train"]["groups"]
    expected_inputs = [df.iloc[fold["train"]["row_positions"]][membership["feature_columns"]].to_numpy()
                       for fold in membership["inner"]]
    expected_inputs.append(df.iloc[membership["outer"]["train"]["row_positions"]][membership["feature_columns"]].to_numpy())
    assert len(scaled_inputs) == len(expected_inputs)
    for actual, expected_input in zip(scaled_inputs, expected_inputs):
        np.testing.assert_array_equal(actual, expected_input)
    model = joblib.load(run / "model.joblib")
    rows = membership["outer"]["train"]["row_positions"]
    expected = df.iloc[rows][membership["feature_columns"]].to_numpy().mean(axis=0)
    np.testing.assert_allclose(model.steps[0][1].mean_, expected)
    assert not np.array_equal(expected, df[membership["feature_columns"]].to_numpy().mean(axis=0))
    if not classifier:
        baseline = json.loads((run / "baseline.json").read_text())
        assert baseline["metrics"] == {"rmse": 0.0, "r2": 1.0}
        test = membership["outer"]["test"]["row_positions"]
        assert baseline["prediction"] == df.iloc[test].target.tolist()


@pytest.mark.parametrize("module", [train, tune])
def test_entrypoints_reject_single_group_before_fit(module, tmp_path, monkeypatch):
    df = frame(); df["N"] = 32
    monkeypatch.setattr(module, "build_dataset", lambda **kwargs: df)
    def forbidden(*args, **kwargs):
        pytest.fail("model search constructed before split rejection")
    monkeypatch.setattr(module, "GridSearchCV", forbidden)
    with pytest.raises(ValueError, match="outer split requires"):
        if module is train:
            module.train_and_evaluate("task_b_kbucket")
        else:
            cfg = tmp_path / "config.yaml"
            cfg.write_text(yaml.safe_dump(dict(task="task_b_kbucket", models=[])))
            module.tune_from_config(cfg)


@pytest.mark.parametrize("module", [train, tune])
def test_explicit_invalid_group_mode_is_not_replaced(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_dataset", lambda **kwargs: frame())
    with pytest.raises(ValueError, match="group mode"):
        if module is train:
            module.train_and_evaluate("task_a_view", group_mode="")
        else:
            cfg = tmp_path / "config.yaml"
            cfg.write_text(yaml.safe_dump(dict(task="task_a_view", group_mode="", models=[])))
            module.tune_from_config(cfg)


def test_native_arithmetic_baseline_uses_only_named_feature():
    y = np.array([4.0, 12.0, 32.0])
    X = np.column_stack([[-999, 0, 999], y])
    result = arithmetic_baseline("task_c_regress_n", X, ["feat_noise", "feat_value_max"], y)
    assert result["prediction"] == y.tolist()
    assert result["metrics"] == {"rmse": 0.0, "r2": 1.0}
    assert arithmetic_baseline("task_a_view", X, [], y) is None


def test_native_generator_baseline_and_usable_task_b_defaults():
    from ns8lab.data import build_dataset, _default_n_values
    from ns8lab.validation import feature_columns
    df = build_dataset("task_c_regress_n", 16, seed=0, n_values=[4, 8, 12], k_values=[1, 2], views=["TLF", "TRB"])
    columns = feature_columns(df)
    result = arithmetic_baseline("task_c_regress_n", df[columns].to_numpy(), columns, df.target.to_numpy())
    assert result["metrics"] == {"rmse": 0.0, "r2": 1.0}
    assert _default_n_values("task_b_kbucket") == (16, 24, 32)
