# UI-UPGRADES.md

Extension plan for the UI, mapped to the current structure and code, with concrete TODOs.

## Current UI structure (files)
- `ui/app.py`: sections for intro/tasks, results snapshot, recent runs, run controls (gated by `mode`), docs tabs, figures gallery, reproduction commands, health/status, sidebar navigation.
- `ui/utils.py`: load UI settings (config/env), parse results table, list configs, index runs, load runs (metrics/config/figures), run commands (subprocess), load text/figures, health info.
- `ui/config.yaml`: mode/paths defaults; env override `NS8_UI_MODE`.
- `requirements-ui.txt`: streamlit dependency.

## Phase UI-1.5 - App shell, navigation, artifact index
Goal: structured navigation + normalized run index with caching.
- [x] Add sidebar navigation/pages (Streamlit radio) for: Browser, Runs, Run Controls, Docs, Figures, Repro, Health.
- [x] Implement `ui/utils.py::index_runs()` to scan `reports/experiments/**` and `runs/**` → normalized list/table with fields: run_id, task, model, timestamp, primary_metric, metrics_path, config_path, cv_path, figures, run_path.
- [x] Define primary metric per task in one place (dict in utils).
- [x] Add caching (`st.cache_data`) for indexing and a “Refresh index” button in UI. (On-disk cache still optional.)

## Phase UI-2.1 - Results Browser v1
Goal: high-signal table and leaderboard.
- [x] Create a "Results Browser" page using the indexed runs table: filters (task/model/date), sort by primary metric.
- [x] Add badges for artifact presence (metrics/config/cv/figures/model).
- [x] Add "Best per task" mini leaderboard at top (per primary metric).
- [x] Graceful fallbacks for missing artifacts.

## Phase UI-2.2 - Run Details page
Goal: trustable single-run view.
- [x] Add Run Details page fed by the index: select run → render metrics.json, config.yaml, cv_results.csv summary, figures thumbnails with expand/fullsize.
- [x] Add CV summary table/plot (e.g., top params/mean scores).
- [x] Notices when artifacts are missing.

## Phase UI-2.3 - Compare Runs
Goal: quick experiment review.
- [x] Add Compare page: select Run A/B from index (same task), show metric deltas and config diffs (only changed keys by default).
- [x] Show side-by-side confusion matrices if available; warn when dataset signatures differ (see Phase UI-4).

## Phase UI-2.4 - Dataset Explorer
Goal: surface data reality.
- [x] Add Dataset Explorer page: load cached dataset sample (or instruct to generate), show head + simple filters, target distributions.
- [x] "Render NS8 grid" for a selected row (N, k, view) by calling grid generator and plotting.
- [x] Optional: a few feature histograms.

## Phase UI-3.1 - Feature importance (when available)
Goal: lightweight interpretability.
- [x] Detect feature_importances_ (RF) and render bar chart; else optional permutation importance (gated/cached) on small slice.
- [x] Save `feature_importance.csv` to the run folder if computed.

## Phase UI-3.2 - Reproduce / Run Jobs page (gated)
Goal: safe run capability.
- [x] Gating via config/env and dry-run preview (basic version done).
- [x] Add live log streaming (not just collected after completion).
- [x] Persist UI logs to run folder (e.g., runs/<id>/ui_stdout.log).
- [x] Add quick buttons: baseline train, tune, tests (with dry-run toggle default on).

## Phase UI-4 - Integrity, fingerprints, polish
Goal: production-ish credibility.
- [x] Compute run fingerprint (hash of config + dataset signature); warn on mismatches.
- [x] Dataset signature check when comparing runs or showing results.
- [x] Health page with versions/paths (basic).
- [x] Persist last filters/run selections more broadly (all pages) via session state.
- [x] Optional: "open folder" hints/links for runs/results.
- [x] Add fingerprint info to health page (placeholder added)

## Function-level TODOs
- `ui/app.py`:
  - Add page routing (sidebar) and split sections into page functions.
  - Add dedicated pages: ResultsBrowser, RunDetails, Compare, DatasetExplorer, Reproduce, Health.
  - Enhance run panel with live log streaming and log persistence.
- `ui/utils.py`:
  - Add helpers: config diff, metrics diff, cv_results summary loader, dataset signature/fingerprint, dataset sample loader, feature importance loader.
  - Add dataset rendering helper (generate grid image from N,k,view).
- `ui/config.yaml`:
  - Add cache path, enable_log_persistence flag, default page, and optional base paths for datasets.

## Notes
- Keep read-only as default mode; run mode opt-in via config/env.
- Prefer cached reads for large folders; add a refresh button when using st.cache_data.
- Maintain artifact path alignment: `runs/` and `reports/experiments/` as primary sources; `reports/figures/` for images; `reports/results/` for summaries.
