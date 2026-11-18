"""Dataset builder for NS8 supervised tasks."""

from __future__ import annotations

from pathlib import Path
import os
from importlib.util import find_spec
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .features import DEFAULT_HIST_BINS, extract_features
from .grids import VALID_VIEWS, generate_grid
from .feature_variants import FEATURE_VARIANTS, c_order_features, pipeline_metadata

TaskName = Literal["task_a_view", "task_b_kbucket", "task_c_regress_n"]


def _default_n_values(task: TaskName) -> Sequence[int]:
    if task == "task_a_view":
        return (16, 24, 32)
    if task == "task_b_kbucket":
        return (16, 24, 32)
    return (16, 24, 32, 48, 64, 96, 128)


def _validate_task(task: str) -> TaskName:
    if task not in {"task_a_view", "task_b_kbucket", "task_c_regress_n"}:
        raise ValueError("task must be one of: task_a_view, task_b_kbucket, task_c_regress_n")
    return task  # type: ignore[return-value]


def _save_dataset(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # pragma: no cover - depends on optional engines
            raise RuntimeError(
                "Writing Parquet requires an installed engine such as pyarrow. "
                "Install an engine or save to .csv instead."
            ) from exc
        return
    raise ValueError(f"Unsupported cache format for {path!s}; use .csv or .parquet")



def _prepare_stable_cache(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing existing stable dataset target: {path}")
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Stable cache requires .csv or .parquet")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Stable cache parent does not exist: {path.parent}")
    if path.suffix.lower() == ".parquet" and not (find_spec("pyarrow") or find_spec("fastparquet")):
        raise RuntimeError("Stable Parquet cache requires an already installed engine; use CSV")
    return path


def _save_stable_dataset(df, path):
    # Preflight rejects existing targets before work; x-mode also closes races.
    created = None
    try:
        csv = path.suffix.lower() == ".csv"
        options = {"encoding": "utf-8", "newline": ""} if csv else {}
        with path.open("x" if csv else "xb", **options) as stream:
            created = os.fstat(stream.fileno())
            if csv:
                df.to_csv(stream, index=False)
            else:
                df.to_parquet(stream, index=False)
    except BaseException:
        if created is not None:
            try:
                current = path.stat()
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    path.unlink()
            except OSError:
                pass
        raise


def build_dataset(
    task: TaskName,
    n_samples: int,
    seed: int = 0,
    n_values: Sequence[int] | None = None,
    k_values: Sequence[int] | None = None,
    views: Sequence[str] | None = None,
    hist_bins: int = DEFAULT_HIST_BINS,
    cache_path: str | Path | None = None,
    k_buckets: int = 8,
    feature_variant: str = "native",
) -> pd.DataFrame:
    """Sample NS8 grids, extract features, and return a tidy DataFrame."""
    task = _validate_task(task)
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError(f"feature_variant must be one of {FEATURE_VARIANTS}")
    stable = feature_variant != "native"
    stable_cache_target = None
    metadata = None
    if stable:
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("Stable datasets require positive integer n_samples")
        stable_cache_target = _prepare_stable_cache(cache_path) if cache_path is not None else None
        metadata = pipeline_metadata(hist_bins)
    rng = np.random.default_rng(seed)

    n_candidates = tuple(n_values) if n_values is not None else _default_n_values(task)
    if not n_candidates:
        raise ValueError("n_values must contain at least one N")
    k_candidates: Iterable[int]
    if k_values is None:
        max_k = max(n_candidates)
        k_candidates = tuple(range(1, max_k + 1))
    else:
        k_candidates = tuple(k_values)
    if not k_candidates:
        raise ValueError("k_values must contain at least one k")
    view_candidates = tuple(views) if views is not None else VALID_VIEWS
    if not view_candidates:
        raise ValueError("views must contain at least one view")
    if k_buckets < 2:
        raise ValueError("k_buckets must be >= 2")

    records = []
    for idx in range(n_samples):
        N = int(rng.choice(n_candidates))
        k = int(rng.choice(k_candidates))
        view = str(rng.choice(view_candidates))
        grid = generate_grid(N, k, view=view)
        feat = c_order_features(grid, hist_bins=hist_bins) if stable else extract_features(grid, hist_bins=hist_bins)

        if task == "task_a_view":
            target = view
            target_type = "classification_view"
        elif task == "task_b_kbucket":
            target = int(k % k_buckets)
            target_type = "classification_k_bucket"
        else:
            target = float(N)
            target_type = "regression_N"

        record = {
            "sample_id": idx,
            "task": task,
            "seed": seed,
            "N": N,
            "k": k,
            "view": view,
            "target": target,
            "target_type": target_type,
        }
        if metadata is not None:
            record.update(metadata)
        record.update(feat)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    if stable and stable_cache_target is not None:
        _save_stable_dataset(df, stable_cache_target)
    elif cache_path is not None:
        _save_dataset(df, Path(cache_path))
    return df


__all__ = ["build_dataset", "TaskName"]
