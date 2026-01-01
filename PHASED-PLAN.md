# PHASED-PLAN.md

## Project Snapshot
- Repo: ns8-supervised-ml-lab
- Pitch: Reproducible supervised-learning benchmark on structured NS8 modular grids, covering dataset generation, preprocessing + pipelines, cross-validation, hyperparameter tuning, and evaluation for classification and regression.
- Why recruiters care: custom synthetic data (not MNIST/Iris), leak-free pipelines, clear ML engineering workflow (data -> features -> model -> metrics -> reproducible experiments).

## Guiding Rules (must keep)
- Follow scikit-learn best practices: train/test split, cross-validation, preprocessing inside Pipelines/ColumnTransformers, hyperparameter tuning, final test only after selection.
- NS8 at the center: implement canonical formulas (A_TRB, A_TLF, plus TRF/BLF/BRF, TLB/BLB/BRB) in `src/ns8lab` with deterministic NumPy grids (values 1..N).
- Reproducibility: fixed seeds, configs versioned, one-command runs (`make setup && make train`), log every run to `runs/` with metrics, params, artifacts.
- Allowed stack: Python 3.11+, numpy, pandas, scikit-learn, matplotlib, joblib; optional rich/pydantic/typer. No deep learning frameworks.

## Targets & Tasks
- Task A (classification): predict view (TLF, TRF, BLF, BRF, TLB, BLB, BRB, TRB) from features.
- Task B (classification): predict kernel bucket (e.g., k mod 8 or quantile bins) with fixed N.
- Task C (regression): predict N across sampled Ns (e.g., 16-128) from summary features.
- Features (initial set): value histograms + entropy/Gini, row/col mean/var, autocorrelation along rows/cols, symmetry scores (H/V/rot180), optional frequency-domain energy.

## Phases

### Phase 0 - Repo foundation (1-2 commits)
**Goal:** Instantly reviewable skeleton.
**Deliverables:**
- Structure: README.md, LICENSE, pyproject.toml (or requirements.txt), .gitignore, src/ns8lab/__init__.py, tests/, notebooks/, data/ (gitignored), reports/figures/.
- README: problem statement, tasks, quickstart, examples, repo map.
- Makefile/justfile commands (e.g., `make train`, `make eval`).
- Minimal unit test that NS8 grid generator matches formulas (spot-check A_TRB/A_TLF).

**Checklist (Phase 0)**
- [x] Base structure: README, LICENSE, pyproject.toml, .gitignore present
- [x] Package scaffold: src/ns8lab/__init__.py and grids module created
- [x] Makefile with setup/test commands
- [x] Placeholder dirs: data/, notebooks/, reports/figures/ tracked via .gitkeep
- [x] Minimal NS8 unit tests for A_TLF/TRB and input validation
- [x] Run test suite (`make test` / `python -m pytest`)
- [ ] Final README polish (examples/usage snippets once code stabilizes)

### Phase 1 - NS8 dataset generator
**Goal:** Math -> dataset factory.
**Deliverables:**
- Grid generation: vectorized NumPy implementation of A_TRB/A_TLF and dihedral transforms (TRF/BLF/BRF, TLB/BLB/BRB); `generate_grid(N, k, view="TLF") -> (N, N)`.
- Task labels: define Tasks A/B/C; sampling strategy for N, k, view; deterministic labeling.
- Features: implement in `src/ns8lab/features.py` per list above.
- Dataset builder: `src/ns8lab/data.py` to sample grids, derive features, emit tidy table with meta (N, k, view, seed) + targets; cache to CSV/Parquet.
- Notebook: `notebooks/01_data_exploration.ipynb` for sanity checks and plots.

**Checklist (Phase 1)**
- [x] Grid generation: complete all dihedral transforms and ensure deterministic `generate_grid`
- [x] Define Tasks A/B/C with sampling strategy for N, k, view; deterministic labels
- [x] Implement feature extraction in `src/ns8lab/features.py` (histograms, entropy/Gini, row/col stats, autocorrelation, symmetry, optional frequency energy)
- [x] Build dataset factory `src/ns8lab/data.py` to sample, featureize, and persist tidy tables with metadata and targets
- [x] Add exploration notebook `notebooks/01_data_exploration.ipynb` with sanity checks and plots
- [x] Wire seeds/configs for reproducibility; consider caching to CSV/Parquet

### Phase 2 - Supervised baselines + evaluation
**Goal:** Clean scikit-learn workflow on tasks.
**Deliverables:**
- Models: LogisticRegression, RandomForestClassifier, SVC/LinearSVC; Ridge/Lasso, RandomForestRegressor.
- Pipelines: StandardScaler for linear/SVM, ColumnTransformer where needed; no preprocessing outside pipelines.
- Validation: train_test_split for hold-out test; cross_val_score or GridSearchCV on train.
- Metrics: classification accuracy + macro F1; regression RMSE + R^2; plots (confusion matrix for view classification, CV score distributions, learning curve) saved to `reports/figures/`.
- Code: `src/ns8lab/train.py`, `src/ns8lab/evaluate.py`, scripts/train_baselines.py.

**Checklist (Phase 2)**
- [x] Implement training pipelines for Task A/B/C with required models (LogReg, RF, SVC/LinearSVC; Ridge/Lasso, RF Regressor)
- [x] Wrap preprocessing inside Pipelines/ColumnTransformers (no leakage); use StandardScaler where appropriate
- [x] Add cross-validation for model selection; keep hold-out test untouched during tuning
- [x] Compute and log metrics (accuracy, macro F1, RMSE, R^2); generate confusion matrix to `reports/figures/`
- [x] Add training/eval entrypoints (`src/ns8lab/train.py`, `src/ns8lab/evaluate.py`, scripts/train_baselines.py`)
- [x] Ensure outputs saved under `runs/` and `reports/` per contract; seeds fixed
- [x] Extend tests to cover pipeline construction and evaluation paths

### Phase 3 - Hyperparameter tuning + experiment tracking
**Goal:** Reproducible experiments.
**Deliverables:**
- Tuning via GridSearchCV (SVM C/gamma, RF depth/estimators, Ridge alpha).
- Central configs (`configs/*.yaml`), seeds fixed.
- Artifacts per run in `reports/experiments/<timestamp>/`: `metrics.json`, `model.joblib`, optional `model_card.md`; save CV results.
- CLI/script: scripts/tune.py; ensure runs also logged under `runs/<timestamp>_<task>_<model>/` per AGENT output contract.

**Checklist (Phase 3)**
- [x] Define central configs (`configs/*.yaml`) with seeds and model grids (SVM C/gamma, RF depth/estimators, Ridge alpha)
- [x] Implement tuning CLI/script (`scripts/tune.py` or similar) that loads configs and writes artifacts
- [x] Persist run artifacts under `runs/<timestamp>_<task>_<model>/` and `reports/experiments/<timestamp>/` (`config.yaml`, `metrics.json`, `model.joblib`, `cv_results.csv`, figures, optional model_card.md)
- [x] Ensure pipelines wrap preprocessing; test set remains untouched during tuning
- [x] Add reporting summaries to `reports/results/` and plots to `reports/figures/`
- [x] Extend tests to cover config loading and tuning output artifacts

### Phase 4 - Recruiter polish + robustness
**Goal:** Feel like a real ML repo.
**Deliverables:**
- Pick one hardening feature: group-aware splits to stop leakage (group by N,k or seed), robustness test on unseen Ns, or feature ablation.
- Minimal CLI: `python -m ns8lab make-dataset ...`, `train ...`, `evaluate ...` (CLI module under src/ns8lab/cli.py).
- Tests: formula correctness (seeds/transforms), no-leakage split test.
- README update: results snapshot, metrics table, design notes (pipelines, CV, tuning, leakage prevention), quickstart.
- Ensure `runs/` and `reports/` contain artifacts from at least two models per task; `make setup && make dataset && make train-classify && make train-regress && make report` target.

**Checklist (Phase 4)**
- [x] Choose and implement one robustness/hardening feature (group-aware splits by N/k)
- [x] Add minimal CLI module (`src/ns8lab/cli.py`) exposing make-dataset/train/tune commands
- [x] Add tests for leakage guards/robustness (group split test) and reinforce formula correctness where needed
- [x] Update README with results snapshot, metrics table, and design notes (pipelines, CV, tuning, leakage prevention); ensure quickstart aligns
- [x] Provide make/just targets for dataset, classification/regression training, and report generation
- [x] Ensure runs/reports include artifacts from at least two models per task; summarize in `reports/results/` (reported in README)

### Phase 5 - Final polish and reproducibility handoff
**Goal:** Publish-ready repo with clear reproduction steps and documented results.
**Tasks:**
- Add a concise results table to `reports/results/` (and/or README) linking recommended configs (e.g., Task B group-N) and metrics.
- Write a short “How to reproduce” section in README covering dataset, train, tune, and artifact locations.
- Add brief model card summaries per task aggregating best models/seeds (reuse existing model_card.md pattern).
- Optional stability check: run Task A with group-mode `n` and include outcomes if stability improves vs `nk`.
- Final tidy pass on docs and CLI instructions; ensure make targets align with README commands.

**Checklist (Phase 5)**
- [x] Populate `reports/results/` (and/or README) with a concise results table referencing recommended configs and metrics
- [x] Add a “How to reproduce” section in README covering dataset/train/tune commands and artifact locations
- [x] Add aggregated model card summaries per task (best models/seeds) using the existing pattern
- [x] Run optional stability check for Task A with group-mode `n` and document outcomes
- [x] Final doc/CLI alignment: ensure README matches make targets and CLI flags; tidy phrasing/formatting

### Phase 6 - Stability and defaults refinement
**Goal:** Improve stability, defaults, and UX.
**Tasks:**
- Stabilize Task A: increase samples and add modest features (e.g., row/col skew/kurtosis or FFT band ratios), rerun seeds 0/1, and update results.
- Make Task B defaults explicit: set `configs/task_b_groupN.yaml` as the documented/default; keep (N,k) grouping as an optional stress test.
- Version pinning: lock `scikit-learn`/`numpy` versions in `pyproject.toml` to reduce API drift and metric variance.
- README visuals: add a brief “figures at a glance” linking to latest confusion matrices in `reports/figures/`.
- CLI defaults: set task-aware `--group-mode` defaults (A/C: nk, B: n) to match recommendations.

**Checklist (Phase 6)**
- [x] Task A stability: add features/increase samples, rerun seeds 0/1, update results table
- [x] Task B defaults: document `configs/task_b_groupN.yaml` as default; keep (N,k) as optional stress test
- [x] Pin core deps (scikit-learn, numpy) in `pyproject.toml`
- [x] Add README “figures at a glance” pointing to latest confusion matrices
- [x] Set task-aware CLI defaults for `--group-mode` (A/C: nk, B: n)
## Acceptance & Output Contract
- Tests pass (`python -m pytest`).
- Every run saves: config.yaml, metrics.json, model.joblib, cv_results.csv (if tuning), figures/ under `runs/<timestamp>_<task>_<model>/` and summaries to `reports/results/` + `reports/figures/`.
- Pipelines wrap preprocessing (no leakage); seeds fixed; metrics printed and logged.
- README and Makefile aligned with commands; quickstart works end-to-end.
