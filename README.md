# ns8-supervised-ml-lab

Reproducible supervised-learning experiments on structured NS8 modular grids. Models train on engineered features (entropy, symmetry, autocorr, FFT) to study invariance, leakage, and scale with transparent artifacts.

## Overview
- Tasks: A (view classification, tests invariance limits), B (k-bucket classification, leakage honesty test via group-aware splits), C (regress N, scale sanity check).
- Data: synthetic NS8 grids → engineered features; no raw grids used for training.
- Transparency: runs log metrics, configs, CV results, models, and figures for inspection.

## Design intent
- Representation over model lottery: data is frozen to engineered features and raw grids are excluded so behavior traces to representation choices instead of memorization or capacity.
- Lightweight models by design: small, efficient estimators make leakage and structure issues visible rather than hiding them behind overpowered models.
- Three diagnostic tasks: Task A probes invariance limits (symmetry vs. representation), Task B spotlights leakage via grouping choices, and Task C sanity-checks whether scale and structure are encoded.

## Quick Start
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest
python -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a.csv
python -m ns8lab.cli train --task task_a_view --samples 400 --seed 0 --group-mode nk
```
UI: `streamlit run ui/app.py` (read-only by default; set `mode: run` or `NS8_UI_MODE=run` to enable run controls).

## UI usage (Streamlit)
- Layout/density: choose Desktop/Tablet/Mobile and Compact/Comfortable in the sidebar.
- Results Browser: click “Apply filters” to refresh; artifacts column shows missing items (hover). Task B grouping caveat noted.
- Figures tab: pick sources (reports/figures, reports/experiments, runs), optional name filter, max-count slider.
- Dataset Explorer: models use engineered features (no raw grids); grid render is for sanity-checking (N, k, view).
- Run Controls: gated by `ui/config.yaml`; commands shown even in read-only mode.

## UI screenshots
- Tasks panel from the Streamlit app with the three workloads (A: view clf, B: k-bucket clf, C: regress N).
![Streamlit tasks panel](media/Tasks.png)

- Overview cards ("What you're seeing") summarizing the engineered-features focus, task intent, and read-only browsing by default.
![Streamlit overview cards](media/What_Your_Seeing.png)

- Results snapshot table filtered to the three tasks, showing grouping, model choices, and metrics/configs from recent runs.
![Streamlit results snapshot](media/Results_Snapshot.png)

## Tasks at a glance
- Task A — View classification: probes invariance limits; errors cluster across mirror-related views.
- Task B — k-bucket classification: tests leakage discipline; strict (N,k) grouping can collapse metrics; group-by-N is recommended for baseline.
- Task C — Regress N: checks that scale is encoded; performs near-perfect when features capture structure.

## Results snapshot (multi-seed)
| Task | Seed | Best model | Metrics | Notes |
| --- | --- | --- | --- | --- |
| Task A (view clf) | 0 | LogisticRegression | acc 0.59, F1 0.51 | reports/experiments/20251231_232904/ (feature tweaks) |
| Task A (view clf) | 1 | LogisticRegression | acc 0.44, F1 0.40 | reports/experiments/20251231_232942/ |
| Task B (k-bucket, group N) | 0/1 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | group-by-N config: `configs/task_b_groupN.yaml` |
| Task B (k-bucket, group N,k) | 1 | all models | acc 0.00, F1 0.00 | strict (N,k) holdout collapses |
| Task C (regress N) | 0/1 | RF regressor | RMSE ~0.0, R^2 ~1.0 | stable across seeds |

## Common commands
- Dataset: `python -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a.csv`
- Train (classify): `python -m ns8lab.cli train --task task_a_view --samples 400 --seed 0 --group-mode nk` (use `--group-mode n` for Task B)
- Train (regress): `python -m ns8lab.cli train --task task_c_regress_n --samples 400 --seed 0`
- Tune: `python -m ns8lab.cli tune --config configs/task_b_groupN.yaml` (Task B recommended), `configs/task_a.yaml`, `configs/task_c.yaml`
- Make targets: `make dataset`, `make train-classify`, `make train-regress`, `make tune`.

## Artifacts
- Per run: `runs/<timestamp>_<task>_<model>/` with `metrics.json`, `config.yaml/json`, `cv_results.csv`, `model.joblib`, `confusion_matrix.png` (classification).
- Mirrored summaries: `reports/experiments/<timestamp>/` and aggregated tables/cards in `reports/results/`.

## Why engineered features (not raw grids)
- Raw grids or deep models would make the tasks trivial via memorization; engineered features constrain what the model can see, making invariance limits, leakage from grouping choices, and scale encoding visible.

## Structure
- `src/ns8lab/` — grids, features, data builder, train/eval pipelines, tuning, CLI.
- `configs/` — YAML configs for tuning tasks.
- `tests/` — unit tests for grids, features, data, training artifacts, tuning.
- `scripts/` — train/tune entry scripts.
- `ui/` — Streamlit app, utils, config, and UI plans (`UI-INFO.md`, `WB-OPT.md`, `UI-FIX.md`, `UI-UPGRADES.md`).
- `reports/results/` — aggregated results/model cards (keep summaries; run-specific artifacts are generated).
- `reports/figures/`, `reports/experiments/` — generated artifacts (gitignored for bulk).
- `data/` — keep small samples; bulk data is gitignored.
- `Makefile`, `pyproject.toml`, `.gitignore`, `LICENSE`, `AGENT-LOG.md`, `AGENT-RULES.md`.

## Cleanup note
- Do not commit bulk artifacts: `runs/`, `reports/experiments/`, `reports/figures/`, large `data/*.csv`.
- Virtualenvs, caches, logs are gitignored; adjust `.gitignore` if you need to track a specific sample.
