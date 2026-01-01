import json

import pytest

from ns8lab import train


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_train_and_evaluate_creates_artifacts(tmp_path):
    # Redirect artifact directories to temp to avoid polluting workspace
    train.RUNS_DIR = tmp_path / "runs"
    train.FIGURES_DIR = tmp_path / "figures"
    train.RESULTS_DIR = tmp_path / "results"

    summaries = train.train_and_evaluate(
        task="task_a_view", n_samples=60, seed=1, test_size=0.25, hist_bins=8, group_mode="nk"
    )

    assert len(summaries) >= 1
    first = summaries[0]
    assert "metrics" in first and "model" in first

    # Check artifacts exist for the first run
    run_id = first["run_id"]
    run_dir = train.RUNS_DIR / run_id
    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "cv_results.csv").exists()
    assert (run_dir / "model.joblib").exists()

    # Metrics file should be valid JSON
    loaded = json.loads((run_dir / "metrics.json").read_text())
    assert "accuracy" in loaded or "rmse" in loaded
