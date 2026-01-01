"""Minimal CLI for dataset creation, training, and tuning."""

from __future__ import annotations

import argparse

from .data import build_dataset
from .train import train_and_evaluate
from .tune import tune_from_config


def _dataset_cmd(args: argparse.Namespace) -> None:
    df = build_dataset(
        task=args.task,
        n_samples=args.n_samples,
        seed=args.seed,
        n_values=args.n_values,
        k_values=args.k_values,
        views=args.views,
        hist_bins=args.hist_bins,
    )
    output = args.output
    if output:
        df.to_csv(output, index=False)
        print(f"Saved dataset to {output} ({len(df)} rows)")
    else:
        print(df.head())


def _train_cmd(args: argparse.Namespace) -> None:
    group_mode = args.group_mode
    if group_mode is None:
        if args.task == "task_b_kbucket":
            group_mode = "n"
        else:
            group_mode = "nk"
    summaries = train_and_evaluate(
        task=args.task,
        n_samples=args.samples,
        seed=args.seed,
        test_size=args.test_size,
        hist_bins=args.hist_bins,
        group_mode=group_mode,
    )
    print(summaries)


def _tune_cmd(args: argparse.Namespace) -> None:
    summaries = tune_from_config(args.config)
    print(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ns8lab", description="NS8 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("dataset", help="Make a sample dataset")
    ds.add_argument("--task", required=True, choices=["task_a_view", "task_b_kbucket", "task_c_regress_n"])
    ds.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    ds.add_argument("--seed", type=int, default=0)
    ds.add_argument("--n-values", type=int, nargs="*", dest="n_values")
    ds.add_argument("--k-values", type=int, nargs="*", dest="k_values")
    ds.add_argument("--views", type=str, nargs="*", dest="views")
    ds.add_argument("--hist-bins", type=int, default=16, dest="hist_bins")
    ds.add_argument("--output", type=str, default=None)
    ds.set_defaults(func=_dataset_cmd)

    tr = sub.add_parser("train", help="Train baseline models")
    tr.add_argument("--task", required=True, choices=["task_a_view", "task_b_kbucket", "task_c_regress_n"])
    tr.add_argument("--samples", type=int, default=400)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    tr.add_argument("--hist-bins", type=int, default=16, dest="hist_bins")
    tr.add_argument(
        "--group-mode",
        type=str,
        default=None,
        choices=["nk", "n", "none"],
        help="Grouping for splits: nk=(N,k), n=by N only, none=no grouping",
    )
    tr.set_defaults(func=_train_cmd)

    tn = sub.add_parser("tune", help="Run tuning from YAML config")
    tn.add_argument("--config", required=True)
    tn.set_defaults(func=_tune_cmd)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
