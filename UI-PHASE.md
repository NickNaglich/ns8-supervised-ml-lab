# UI-PHASE.md

Phased workflow for a simple but useful Streamlit UI to explain, showcase, and operate the NS8 ML repo. Focus on results browsing, quick runs, and credibility.

## Phase UI-0 — Scope & skeleton
- Define persona: recruiter/engineer wants to see what was trained, metrics, and how to reproduce; optionally trigger lightweight runs.
- Decide constraints: read-only by default (browse artifacts); optional “run” mode gated by a flag.
- Skeleton Streamlit app (`ui/app.py`) with sections: intro, tasks overview, recent runs, artifacts browser.
- Run-mode gating: default `mode: read-only` in `ui/config.yaml`; switch to `run` (or via env override) if enabling run controls later.

**Checklist (UI-0)**
- [x] Confirm UI scope/persona and default to read-only mode
- [x] Decide run-mode gating approach (flag/env) and document it
- [x] Scaffold `ui/app.py` with sections placeholders (intro, tasks overview, recent runs, artifacts browser)
- [x] Add `ui/utils.py` stub and `ui/config.yaml` placeholder for paths/mode
- [x] Add optional UI dependency list (streamlit) in a separate requirements file or extras

## Phase UI-1 — Artifact browser & results
- Parse `reports/experiments/*/*/metrics.json`, `cv_results.csv`, `config.yaml`, `model_card.md`, and figures.
- Show per-task summary tables (best run per task, metrics, config link, confusion matrix thumbnail).
- Add filters: by task, model, timestamp; sort by metric.
- Provide “open in explorer” links to run folders and figures.

**Checklist (UI-1)**
- [x] Parse results snapshot and render summary tables in the UI
- [x] Load recent runs from `reports/experiments/` with metrics, configs, and figure thumbnails
- [x] Add filters (task/model/timestamp) and sorting by metric in the browser
- [x] Add links/buttons to open run folders/figures (path display)
- [x] Handle missing artifacts gracefully with notices

## Phase UI-2 — Run controls (optional/gated)
- Add a “Run a job” panel (behind a toggle/checkbox) to trigger:
  - `tune` via chosen config (dropdown of `configs/*.yaml`)
  - `train` quick baseline with small samples
  - `tests` (`python -m pytest`) for verification
- Show live logs in the UI (stream subprocess output); disable by default in shared environments.
- Add guardrails: confirm dialog; optional “dry run” to just show the command.

**Checklist (UI-2)**
- [x] Add a gated “Run a job” panel toggled by mode (read-only vs run)
- [x] Populate config dropdowns and train/test presets; allow dry-run preview of commands
- [x] Stream subprocess output/logs to the UI (with run disabled by default)
- [x] Add confirmations and warnings (resource usage, path safety)
- [x] Ensure commands respect grouping defaults (Task B group-N, etc.)

## Phase UI-3 — Explainability & docs surfacing
- Embed key docs: README snippets, PHASED-PLAN highlights, model cards.
- Add “figures at a glance”: grid of confusion matrices from `reports/figures/`.
- Add a short “How to reproduce” card with copyable commands (dataset/train/tune).

**Checklist (UI-3)**
- [x] Add README/PHASED-PLAN/model card excerpts into the UI (tabs or expanders)
- [x] Add a “figures at a glance” gallery pulling from `reports/figures/`
- [x] Add a “How to reproduce” card with copyable commands (dataset/train/tune)
- [x] Provide links to results/model_cards in `reports/results/`
- [x] Ensure missing-file handling (show notices if docs/figures absent)

## Phase UI-4 — Production polish
- Add config for paths and modes (read-only vs run-enabled) via `ui/config.yaml`.
- Persist user selections (last viewed task/run) in session state.
- Handle missing artifacts gracefully; show alerts for outdated deps.
- Provide a simple health check (deps/version info, pinned versions).

**Checklist (UI-4)**
- [x] Move mode/paths into `ui/config.yaml` with clear defaults and env override
- [x] Persist UI selections (task filters, run filters) in session state
- [x] Add health/status info (deps versions, pinned versions, path checks)
- [x] Improve error handling/alerts for missing artifacts or bad configs
- [ ] Optional: link to runs/results folders with open-in-file-explorer hints

## Nice-to-have
- Compare runs: select two runs to diff metrics/params.
- Lightweight dataset preview: show head of a generated dataset (cached) and feature importances if available.

## File/layout plan
- `ui/app.py` — Streamlit entrypoint with sections and routing.
- `ui/utils.py` — helpers to load artifacts, parse metrics/configs, render figures.
- `ui/config.yaml` — UI settings (mode, paths).
- `requirements-ui.txt` (or optional extra) — add `streamlit` only for UI.

## Milestones
- UI-0: skeleton app with intro + placeholder sections.
- UI-1: artifact browser live with tables/figures.
- UI-2: gated run controls and test trigger.
- UI-3: docs/model cards embedded; figures gallery.
- UI-4: polish (config, error handling, version/health info).

## Safety defaults
- Default to read-only mode; require explicit flag to run jobs.
- Never delete artifacts; only read/browse.
- Keep runs small if triggered from UI; point to configs for full runs.
