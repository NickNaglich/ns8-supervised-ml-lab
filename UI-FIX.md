# UI-FIX.md

Phased fixes to align the UI with the suggested widgets and visuals, mapped to this project’s structure.

## Phase UI-FX0 — Inventory & Wiring
- [x] Verify current pages load (streamlit run ui/app.py); note any crashes.
- [x] List existing artifacts per task (metrics, confusion matrices, figures) to know what can be visualized.
- [x] Add placeholder tabs/cards in `ui/app.py` for Dataset, Model Visuals, Experiment Browser enhancements.

## Phase UI-FX1 — Dataset Overview
- [x] Dataset head: scrollable table (use `st.dataframe` with `use_container_width=True`).
- [x] Feature histograms for key engineered features (entropy, symmetry, FFT bands) with multi-select.
- [x] Class/label balance bar chart for classification tasks (A/B) when `target` present.
- [x] Wire to cached dataset sample loader in `ui/utils.py`; add clear “load/generate sample dataset” helper text.

## Phase UI-FX2 — Model Performance Visuals
- [x] Confusion matrix heatmap widget: render PNG when present (Run Details).
- [x] ROC / PR curves: render from saved figures if present; if absent and data available, compute lightweight curves (gated).
- [x] Metric trend plots: load `cv_results.csv` when present and plot metric vs hyperparams; fallback notice if missing.
- [x] Add thumbnails for figures (confusion matrices, learning curves) in results browser rows.

## Phase UI-FX3 — Experiment Browser Upgrades
- [x] Filters: task/model/metric-threshold sliders in Results Browser.
- [x] Sortable table with primary metric; add quick “Top N” toggle.
- [x] Artifact badges already present: add hover/tooltips for what’s missing (textual hints).
- [x] Add “open folder” hints/links for run directories (run_path column + caption).

## Phase UI-FX4 — Run Compare Enhancements
- [x] Two-run selector persists in session; enforce same task.
- [x] Delta bar chart for selected metric (F1/Acc/RMSE/R2) with highlight.
- [x] Config diff already present: add toggle to show unchanged keys.
- [x] Figures side-by-side: confusion matrices and PR/ROC when available; show “missing” badges otherwise.
- [x] Dataset signature warning already added: extend with explicit badge in compare view.

## Phase UI-FX5 — Interactive Charts & Drill-downs
- [x] Line chart: metric vs hyperparameter from `cv_results.csv` (select metric/param).
- [x] Scatter/pair plot option for a small sampled dataset (gated, cached).
- [x] Click-to-expand behavior: clicking a run row shows its figures/metrics inline.

## Phase UI-FX6 — Polish & Reliability
- [x] Add “refresh index” + “recompute curves” buttons with cache clearing where needed.
- [x] Graceful fallbacks everywhere: explicit text when artifacts or datasets are missing.
- [x] Light theme consistency: captions for every chart, units/metric names.
- [x] Optional: persist rendered feature importance to `feature_importance.csv` (already supported) and surface download link.

Notes:
- Keep run mode gated via `ui/config.yaml` / `NS8_UI_MODE`.
- Prefer existing loaders (`index_runs`, `load_run_details`, `load_dataset_sample`) and cache with `st.cache_data`.
- Place visual helpers in `ui/utils.py` when logic is non-trivial (e.g., ROC/PR computation, confusion matrix plotting).
