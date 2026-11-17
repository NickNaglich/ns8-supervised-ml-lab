"""Training and evaluation entrypoints for NS8 tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

from .data import TaskName, build_dataset
from .evaluate import classification_metrics, plot_confusion_matrix, regression_metrics, summarize_cv

from .validation import arithmetic_baseline, prepare_split
from .validation import build_groups as _build_groups
from .validation import feature_columns as _feature_columns

FIGURES_DIR = Path("reports/figures")
RUNS_DIR = Path("runs")
RESULTS_DIR = Path("reports/results")


def _split(df: pd.DataFrame, task: TaskName, test_size: float, seed: int, group_mode: str) -> Tuple:
    """Compatibility tuple; actual entrypoints retain the full split contract."""
    split = prepare_split(df, task, test_size, seed, group_mode)
    return split.X[split.train], split.X[split.test], split.y[split.train], split.y[split.test]


def _make_classifiers(seed: int) -> Dict[str, Tuple[Pipeline, Dict]]:
    preprocess = ColumnTransformer([("num", StandardScaler(), slice(0, None))])
    models: Dict[str, Tuple[Pipeline, Dict]] = {
        "logreg": (
            Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))]),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
        "linear_svc": (
            Pipeline([("prep", preprocess), ("clf", LinearSVC(random_state=seed))]),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
        "rf_clf": (
            Pipeline([("clf", RandomForestClassifier(n_estimators=150, max_depth=None, random_state=seed))]),
            {"clf__max_depth": [None, 8, 16]},
        ),
    }
    return models


def _make_regressors(seed: int) -> Dict[str, Tuple[Pipeline, Dict]]:
    preprocess = ColumnTransformer([("num", StandardScaler(), slice(0, None))])
    models: Dict[str, Tuple[Pipeline, Dict]] = {
        "ridge": (
            Pipeline([("prep", preprocess), ("reg", Ridge(random_state=seed))]),
            {"reg__alpha": [0.1, 1.0, 10.0]},
        ),
        "lasso": (
            Pipeline([("prep", preprocess), ("reg", Lasso(max_iter=5000, random_state=seed))]),
            {"reg__alpha": [0.001, 0.01, 0.1]},
        ),
        "rf_reg": (
            Pipeline([("reg", RandomForestRegressor(n_estimators=200, random_state=seed))]),
            {"reg__max_depth": [None, 10, 20]},
        ),
    }
    return models


def _metric_for_task(task: TaskName):
    if task in {"task_a_view", "task_b_kbucket"}:
        return "f1_macro"
    return "r2"


def _labels_for_task(task: TaskName, df: pd.DataFrame):
    if task in {"task_a_view", "task_b_kbucket"}:
        return sorted(df["target"].unique())
    return None


def train_and_evaluate(
    task: TaskName,
    n_samples: int = 400,
    seed: int = 0,
    test_size: float = 0.2,
    hist_bins: int = 16,
    group_mode: str | None = None,
):
    df = build_dataset(task=task, n_samples=n_samples, seed=seed, hist_bins=hist_bins)
    if group_mode is None:
        group_mode = "n" if task == "task_b_kbucket" else "nk"
    split = prepare_split(df, task, test_size, seed, group_mode)
    X_train, X_test = split.X[split.train], split.X[split.test]
    y_train, y_test = split.y[split.train], split.y[split.test]
    cv = split.cv
    baseline = arithmetic_baseline(task, X_test, split.provenance["feature_columns"], y_test)
    models = _make_classifiers(seed) if task in {"task_a_view", "task_b_kbucket"} else _make_regressors(seed)
    scoring = _metric_for_task(task)

    results = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for model_name, (pipe, param_grid) in models.items():
        search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train, groups=split.train_groups)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        if task in {"task_a_view", "task_b_kbucket"}:
            metrics_dict = classification_metrics(y_test, y_pred)
        else:
            metrics_dict = regression_metrics(y_test, y_pred)

        run_id = f"{timestamp}_{task}_{model_name}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        (run_dir / "split.json").write_text(json.dumps(split.provenance, indent=2))
        if baseline is not None:
            (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_dict, indent=2))

        config_path = run_dir / "config.json"
        config_payload = {
            "task": task,
            "n_samples": n_samples,
            "seed": seed,
            "test_size": test_size,
            "hist_bins": hist_bins,
            "model": model_name,
            "group_mode": group_mode,
            "cv_splits": len(split.provenance["inner"]),
            "param_grid": param_grid,
        }
        config_path.write_text(json.dumps(config_payload, indent=2))

        cv_df = summarize_cv(search.cv_results_)
        cv_df.to_csv(run_dir / "cv_results.csv", index=False)

        joblib.dump(best_model, run_dir / "model.joblib")

        labels = _labels_for_task(task, df)
        if labels is not None:
            fig, _ = plot_confusion_matrix(y_test, y_pred, labels=labels, title=f"{task} - {model_name}")
            fig_path = run_dir / "confusion_matrix.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            # also copy to reports/figures
            shutil.copyfile(fig_path, FIGURES_DIR / fig_path.name)

        summary = {
            "run_id": run_id,
            "task": task,
            "model": model_name,
            "metrics": metrics_dict,
            "best_params": search.best_params_,
            "group_mode": group_mode,
            "baseline": baseline,
        }
        (RESULTS_DIR / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2))

        results.append(summary)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NS8 baselines")
    parser.add_argument("--task", choices=["task_a_view", "task_b_kbucket", "task_c_regress_n"], required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hist-bins", type=int, default=16)
    args = parser.parse_args()

    summaries = train_and_evaluate(
        task=args.task,
        n_samples=args.samples,
        seed=args.seed,
        test_size=args.test_size,
        hist_bins=args.hist_bins,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
