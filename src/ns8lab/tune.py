"""Hyperparameter tuning and experiment tracking for NS8 tasks."""

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
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import TaskName, build_dataset
from .evaluate import classification_metrics, plot_confusion_matrix, regression_metrics, summarize_cv

matplotlib.use("Agg")

from .validation import arithmetic_baseline, prepare_split
from .validation import build_groups as _build_groups
from .validation import feature_columns as _feature_columns

RUNS_DIR = Path("runs")
EXPERIMENTS_DIR = Path("reports/experiments")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")


def _split(df: pd.DataFrame, task: TaskName, test_size: float, seed: int, group_mode: str) -> Tuple:
    """Compatibility tuple; actual entrypoints retain the full split contract."""
    split = prepare_split(df, task, test_size, seed, group_mode)
    return split.X[split.train], split.X[split.test], split.y[split.train], split.y[split.test]


def _prepare_grid(raw_grid: Dict[str, List], step: str) -> Dict[str, List]:
    return {k if "__" in k else f"{step}__{k}": v for k, v in raw_grid.items()}


def _model_registry(seed: int, task: TaskName) -> Dict[str, Tuple[Pipeline, str]]:
    preprocess = ColumnTransformer([("num", StandardScaler(), slice(0, None))])
    registry: Dict[str, Tuple[Pipeline, str]] = {
        "logreg": (
            Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))]),
            "clf",
        ),
        "svm_rbf": (
            Pipeline([("prep", preprocess), ("clf", SVC())]),
            "clf",
        ),
        "rf_clf": (
            Pipeline([("clf", RandomForestClassifier(random_state=seed))]),
            "clf",
        ),
        "ridge": (
            Pipeline([("prep", preprocess), ("reg", Ridge(random_state=seed))]),
            "reg",
        ),
        "rf_reg": (
            Pipeline([("reg", RandomForestRegressor(random_state=seed))]),
            "reg",
        ),
    }
    # Filter registry based on task type
    if task in {"task_a_view", "task_b_kbucket"}:
        return {k: v for k, v in registry.items() if k in {"logreg", "svm_rbf", "rf_clf"}}
    return {k: v for k, v in registry.items() if k in {"ridge", "rf_reg"}}


def _dump_yaml(obj: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def _make_model_card(task: TaskName, model_name: str, metrics_dict: Dict[str, float]) -> str:
    lines = [
        f"# Model Card - {task} - {model_name}",
        "",
        "## Metrics",
    ]
    for k, v in metrics_dict.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Tuned with GridSearchCV; metrics reported on hold-out test split.")
    return "\n".join(lines)


def tune_from_config(config_path: str | Path, group_mode: str | None = None):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    task: TaskName = config["task"]
    seed = int(config.get("seed", 0))
    test_size = float(config.get("test_size", 0.2))
    hist_bins = int(config.get("hist_bins", 16))
    n_samples = int(config.get("n_samples", 400))
    n_values = config.get("n_values")
    k_values = config.get("k_values")
    views = config.get("views")
    if group_mode is None:
        group_mode = "n" if task == "task_b_kbucket" else "nk"
    group_mode = config.get("group_mode", group_mode)
    k_buckets = int(config.get("k_buckets", 8))
    cv_splits = config.get("cv_splits", 3)
    models_cfg = config.get("models", [])

    df = build_dataset(
        task=task,
        n_samples=n_samples,
        seed=seed,
        n_values=n_values,
        k_values=k_values,
        views=views,
        hist_bins=hist_bins,
        k_buckets=k_buckets,
    )
    split = prepare_split(df, task, test_size, seed, group_mode, cv_splits)
    X_train, X_test = split.X[split.train], split.X[split.test]
    y_train, y_test = split.y[split.train], split.y[split.test]
    cv = split.cv
    baseline = arithmetic_baseline(task, X_test, split.provenance["feature_columns"], y_test)
    config = dict(config, group_mode=group_mode, cv_splits=len(split.provenance["inner"]))

    registry = _model_registry(seed, task)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summaries = []

    for model_entry in models_cfg:
        model_name = model_entry["name"]
        param_grid_raw = model_entry.get("param_grid", {})
        if model_name not in registry:
            raise ValueError(f"Unknown model '{model_name}' for task {task}")
        pipeline, step_name = registry[model_name]
        param_grid = _prepare_grid(param_grid_raw, step=step_name)

        scoring = "f1_macro" if task in {"task_a_view", "task_b_kbucket"} else "r2"

        search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train, groups=split.train_groups)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        if task in {"task_a_view", "task_b_kbucket"}:
            metrics_dict = classification_metrics(y_test, y_pred)
        else:
            metrics_dict = regression_metrics(y_test, y_pred)

        run_id = f"{timestamp}_{task}_{model_name}_tune"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        exp_dir = EXPERIMENTS_DIR / timestamp / model_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        (run_dir / "split.json").write_text(json.dumps(split.provenance, indent=2))
        if baseline is not None:
            (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_dict, indent=2))
        shutil.copyfile(metrics_path, exp_dir / "metrics.json")

        config_out = run_dir / "config.yaml"
        _dump_yaml(config, config_out)
        shutil.copyfile(config_out, exp_dir / "config.yaml")

        cv_df = summarize_cv(search.cv_results_)
        cv_path = run_dir / "cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        shutil.copyfile(cv_path, exp_dir / "cv_results.csv")

        model_path = run_dir / "model.joblib"
        joblib.dump(best_model, model_path)
        shutil.copyfile(model_path, exp_dir / "model.joblib")

        model_card = _make_model_card(task, model_name, metrics_dict)
        (exp_dir / "model_card.md").write_text(model_card)

        labels = sorted(df["target"].unique()) if task in {"task_a_view", "task_b_kbucket"} else None
        if labels is not None:
            fig, _ = plot_confusion_matrix(y_test, y_pred, labels=labels, title=f"{task} - {model_name} (tune)")
            fig_path = run_dir / "confusion_matrix.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            shutil.copyfile(fig_path, exp_dir / "confusion_matrix.png")
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
        summaries.append(summary)

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Tune NS8 models via config")
    parser.add_argument("--config", required=True, help="Path to tuning config YAML")
    args = parser.parse_args()
    summaries = tune_from_config(args.config)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
