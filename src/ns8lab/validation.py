"""Explicit outer/inner split contracts shared by train and tune.

Grouping is an experiment decision. Infeasible decisions fail before any fit;
they never silently become row splits. Artifacts retain original row positions.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

CLASSIFICATION = {"task_a_view", "task_b_kbucket"}


def feature_columns(df):
    return [c for c in df.columns if c.startswith("feat_")]


def build_groups(df, mode):
    if mode == "none":
        return None
    if mode == "n":
        return df["N"].to_numpy()
    if mode == "nk":
        # generate_grid reduces slope modulo N; group aliases together too.
        return np.array(
            [f"{int(n)}_{int(k) % int(n)}" for n, k in zip(df["N"], df["k"])],
            dtype=object,
        )
    raise ValueError("group mode must be one of: none, n, nk")


def _check_partition(task, y, train, valid, context):
    minimum = 1 if task in CLASSIFICATION else 2
    if len(train) < minimum or len(valid) < minimum:
        raise ValueError(f"{context}: insufficient train/validation rows")
    if task in CLASSIFICATION:
        labels = set(y.tolist())
        if len(labels) < 2 or set(y[train].tolist()) != labels or set(y[valid].tolist()) != labels:
            raise ValueError(f"{context}: every class must occur in both partitions")


@dataclass
class PreparedSplit:
    X: np.ndarray
    y: np.ndarray
    train: np.ndarray
    test: np.ndarray
    groups: np.ndarray | None
    cv: object
    provenance: dict

    @property
    def train_groups(self):
        return None if self.groups is None else self.groups[self.train]


def prepare_split(df, task, test_size, seed, group_mode, requested_splits=3):
    if task not in CLASSIFICATION | {"task_c_regress_n"}:
        raise ValueError("unsupported supervised task")
    if type(requested_splits) is not int or requested_splits < 2:
        raise ValueError("cv_splits must be an integer >= 2")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    columns = feature_columns(df)
    if not columns or df.empty:
        raise ValueError("nonempty dataset and feat_ columns required")
    X = df[columns].to_numpy(dtype=float)
    y = df["target"].to_numpy()
    if not np.isfinite(X).all() or df["target"].isna().any():
        raise ValueError("features and targets must be finite/nonmissing")
    if task not in CLASSIFICATION and not np.isfinite(y.astype(float)).all():
        raise ValueError("regression targets must be finite")
    groups = build_groups(df, group_mode)
    rows = np.arange(len(df))
    if groups is not None:
        if len(np.unique(groups)) < 2:
            raise ValueError("outer split requires at least two distinct groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(X, y, groups))
    else:
        if task in CLASSIFICATION:
            _, counts = np.unique(y, return_counts=True)
            if len(counts) < 2 or counts.min() < 2:
                raise ValueError("outer split requires at least two examples per class")
        train, test = train_test_split(
            rows, test_size=test_size, random_state=seed,
            stratify=y if task in CLASSIFICATION else None,
        )
    _check_partition(task, y, train, test, "outer split")
    yt = y[train]
    gt = None if groups is None else groups[train]
    if gt is not None:
        group_count = len(np.unique(gt))
        if group_count < 2:
            raise ValueError("inner CV requires at least two training groups")
        cv = GroupKFold(n_splits=min(requested_splits, group_count))
    elif task in CLASSIFICATION:
        _, counts = np.unique(yt, return_counts=True)
        if counts.min() < 2:
            raise ValueError("inner CV requires at least two training examples per class")
        cv = StratifiedKFold(n_splits=min(requested_splits, int(counts.min())), shuffle=True, random_state=seed)
    else:
        if len(train) < requested_splits:
            raise ValueError("inner CV has fewer rows than requested folds")
        cv = KFold(n_splits=requested_splits, shuffle=True, random_state=seed)

    def membership(indices):
        return {
            "row_positions": indices.tolist(),
            "sample_ids": df.iloc[indices]["sample_id"].tolist() if "sample_id" in df else indices.tolist(),
            "groups": None if groups is None else groups[indices].tolist(),
        }

    folds = []
    for fold_train, fold_valid in cv.split(X[train], yt, gt):
        _check_partition(task, yt, fold_train, fold_valid, "inner CV")
        if gt is not None and set(gt[fold_train]) & set(gt[fold_valid]):
            raise ValueError("inner CV group overlap")
        folds.append({"train": membership(train[fold_train]), "validation": membership(train[fold_valid])})
    if groups is not None and set(groups[train]) & set(groups[test]):
        raise ValueError("outer split group overlap")
    provenance = {
        "schema_version": 1, "group_mode": group_mode,
        "group_definition": "N,k modulo N" if group_mode == "nk" else group_mode,
        "feature_columns": columns, "outer": {"train": membership(train), "test": membership(test)},
        "inner_splitter": type(cv).__name__, "inner": folds,
    }
    return PreparedSplit(X, y, train, test, groups, cv, provenance)


def arithmetic_baseline(task, X, columns, y):
    """No fitted model: the generator directly exposes N as its maximum value."""
    if task != "task_c_regress_n":
        return None
    from .evaluate import regression_metrics

    prediction = X[:, columns.index("feat_value_max")]
    return {
        "name": "value_max_identity", "prediction": prediction.tolist(),
        "metrics": regression_metrics(y, prediction),
        "scope": "Synthetic additive complete-square generator exposes N; not learned generalization.",
    }
