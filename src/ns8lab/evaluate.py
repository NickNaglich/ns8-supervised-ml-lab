"""Evaluation utilities for NS8 supervised tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray | None = None


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1)}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    # squared=False is not available in older sklearn builds; compute RMSE manually.
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2)}


def plot_confusion_matrix(y_true, y_pred, labels, title: str) -> Tuple[plt.Figure, plt.Axes]:
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def summarize_cv(cv_results: dict) -> pd.DataFrame:
    """Convert GridSearchCV.cv_results_ to a tidy DataFrame."""
    return pd.DataFrame(cv_results)


__all__ = ["EvalResult", "classification_metrics", "regression_metrics", "plot_confusion_matrix", "summarize_cv"]
