# ns8-supervised-ml-lab

Reproducible supervised-learning benchmark on structured NS8 modular grids. Shows dataset generation, preprocessing with pipelines, cross-validation, hyperparameter tuning, and evaluation for both classification and regression - built to be recruiter-friendly and leak-free.

## Project intent
- Keep NS8 math central (TRB/TLF kernels and derived transforms).
- Demonstrate scikit-learn fundamentals: pipelines, CV, tuning, clean splits, reproducible runs.
- Ship artifacts and summaries a recruiter can read quickly, with transparency over raw scores.
- Use engineered features (entropy, symmetry, autocorr, FFT) instead of raw grids to study invariance, leakage, and scale.

## Quickstart (Phase 0)
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest
```
Or with make:
```bash
make setup
make test
```

## Roadmap snapshot
- Phase 0: repo skeleton + basic formulas test (this commit).
- Phase 1: NS8 grid generator, features, dataset builder.
- Phase 2: supervised baselines + evaluation (classification and regression).
- Phase 3: hyperparameter tuning and experiment tracking.
- Phase 4: polish, leakage guards, CLI, results and figures.

## Dataset generation (Phase 1 in progress)
```bash
python - <<'PY'
from ns8lab.data import build_dataset
df = build_dataset(task="task_a_view", n_samples=5, seed=0)
print(df.head())
PY
```

## Training baselines (Phase 2)
```bash
python -m ns8lab.train --task task_a_view --samples 400 --seed 0
# or
python scripts/train_baselines.py --task task_c_regress_n --samples 400 --seed 0
```
Artifacts land in `runs/<timestamp>_<task>_<model>/` with `metrics.json`, `config.json`, `cv_results.csv`, `model.joblib`, and figures (confusion matrix for classification). Summaries are also written to `reports/results/` and figures to `reports/figures/`.

## Hyperparameter tuning (Phase 3)
Use the YAML configs under `configs/` to drive tuning runs and artifact logging:
```bash
python scripts/tune.py --config configs/task_a.yaml
```
Artifacts are mirrored to `runs/` and `reports/experiments/<timestamp>/` (metrics, config, cv_results, model, confusion matrix if applicable, and a simple model card).

## How to reproduce (Phase 5)
1) Install deps: `python -m pip install -e .[dev]`
2) Make a dataset: `python -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a.csv`
3) Train baselines: `python -m ns8lab.cli train --task task_a_view --samples 400 --seed 0`
4) Tune from config: `python -m ns8lab.cli tune --config configs/task_b_groupN.yaml`
Artifacts: per-run folders in `runs/<timestamp>_<task>_<model>/`; mirrored summaries in `reports/experiments/<timestamp>/`, figures in `reports/figures/`, and aggregated results in `reports/results/`.

### CLI shortcuts
- Dataset: `python -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a.csv`
- Train classify: `python -m ns8lab.cli train --task task_a_view --samples 400 --seed 0 --group-mode nk` (use `--group-mode n` for Task B if avoiding k-holdout)
- Train regress: `python -m ns8lab.cli train --task task_c_regress_n --samples 400 --seed 0`
- Tune: `python -m ns8lab.cli tune --config configs/task_b_groupN.yaml` (Task B recommended), `configs/task_a.yaml`, `configs/task_c.yaml`

Make targets mirror these: `make dataset`, `make train-classify`, `make train-regress`, `make tune`.

## Current results (multi-seed snapshot)
| Task | Seed | Best model | Metrics | Notes |
| --- | --- | --- | --- | --- |
| Task A (view clf) | 0 | LogisticRegression | acc 0.59, F1 0.51 | reports/experiments/20251231_232904/ (after feature tweaks) |
| Task A (view clf) | 1 | LogisticRegression | acc 0.44, F1 0.40 | reports/experiments/20251231_232942/ |
| Task B (k-bucket, group N) | 0 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | reports/experiments/20251231_230644_* (group by N only) |
| Task B (k-bucket, group N) | 1 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | reports/experiments/20251231_230703_* (group by N only) |
| Task B (k-bucket, group N,k) | 0 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | reports/experiments/20251231_224707/ |
| Task B (k-bucket, group N,k) | 1 | (all models) | acc 0.00, F1 0.00 | grouping by (N,k) held out k buckets |
| Task C (regress N) | 0 | RF regressor | RMSE ~0.0, R² ~1.0 | reports/experiments/20251231_224946/ |
| Task C (regress N) | 1 | RF regressor | RMSE ~0.0, R² ~1.0 | reports/experiments/20251231_230503/ |

Seed 1 runs used larger samples and group-aware splits. Task B is stable when grouping by N only; grouping by (N,k) can zero-out performance (holds out k buckets). Use `configs/task_b_groupN.yaml` as the recommended config.

## Streamlit UI (read-only by default)
- Launch: `streamlit run ui/app.py`
- Layout selector: pick Desktop/Tablet/Mobile in the sidebar; density toggle (Compact/Comfortable) adjusts spacing/fonts.
- Results Browser: filters are applied via an “Apply filters” button; artifacts column shows missing items (hover). Task B grouping caveat is noted.
- Figures tab: choose sources (reports/figures, reports/experiments, runs), optional name filter, and a max-count slider.
- Dataset Explorer: reminder that models use engineered features (no raw grids); grid render is for sanity-checking (N, k, view).
- Run controls stay gated by `ui/config.yaml` (`mode: read-only` by default); set `mode: run` or `NS8_UI_MODE=run` to enable.

## Structure (Phase 3)
- README.md, LICENSE, pyproject.toml, .gitignore
- src/ns8lab/ - grids, feature extraction, dataset builder, training/eval pipelines, tuning
- configs/ - YAML configs for tuning tasks
- tests/ - unit tests for grids, features, dataset factory, training artifacts, tuning
- scripts/train_baselines.py and scripts/tune.py - CLI entries
- notebooks/01_data_exploration.ipynb - exploration placeholder
- data/ - raw and processed data (gitignored; keep small samples only)
- reports/figures/, reports/results/, reports/experiments/ - generated plots and summaries (gitignored for bulk artifacts)
- ui/ - Streamlit app, utils, config, and UI plans (WB-OPT.md, UI-INFO.md track UX/narrative work)

## Cleanup before push
- Keep code, configs, small sample data; keep results summaries in `reports/results/`.
- Do not commit bulk artifacts: `runs/`, `reports/experiments/`, `reports/figures/`, large `data/*.csv` (gitignored).
- Virtualenvs, caches, logs are gitignored; adjust `.gitignore` if you need to track a specific sample.
