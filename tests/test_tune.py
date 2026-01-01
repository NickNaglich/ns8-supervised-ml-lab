import json

import yaml

from ns8lab import tune


def test_tune_from_config_creates_artifacts(tmp_path):
    # Redirect artifact directories to temp
    tune.RUNS_DIR = tmp_path / "runs"
    tune.EXPERIMENTS_DIR = tmp_path / "experiments"
    tune.FIGURES_DIR = tmp_path / "figures"
    tune.RESULTS_DIR = tmp_path / "results"

    cfg = {
        "task": "task_a_view",
        "seed": 1,
        "test_size": 0.25,
        "hist_bins": 8,
        "n_samples": 80,
        "n_values": [4],
        "k_values": [1, 2],
        "views": ["TLF", "TRB"],
        "cv_splits": 2,
        "models": [{"name": "logreg", "param_grid": {"C": [1.0]}}],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    summaries = tune.tune_from_config(cfg_path, group_mode="nk")
    assert len(summaries) == 1

    summary = summaries[0]
    run_dir = tune.RUNS_DIR / summary["run_id"]
    assert run_dir.exists()
    for fname in ["metrics.json", "config.yaml", "cv_results.csv", "model.joblib"]:
        assert (run_dir / fname).exists()

    # Metrics should be valid JSON
    loaded = json.loads((run_dir / "metrics.json").read_text())
    assert "accuracy" in loaded or "rmse" in loaded

    # Experiment copy should exist
    timestamp = "_".join(summary["run_id"].split("_", 2)[:2])
    exp_dir = tune.EXPERIMENTS_DIR / timestamp / summary["model"]
    assert exp_dir.exists()
    assert (exp_dir / "model_card.md").exists()
