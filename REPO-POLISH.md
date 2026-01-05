Phase 0 — Repo polish + trust signals (CI/quality gates)

Goal: Make the repo look “production-serious” in 30 seconds.

Tasks

Add GitHub Actions workflow:

pytest on Python 3.11 (and optionally 3.12)

cache pip

Add lint/format:

ruff (lint + format)

optional mypy for src/ns8lab/* (lightweight)

Add pre-commit config running ruff + end-of-file-fixer.

Add badges to README: CI status, Python version, license.

Ensure make test, make lint, make format exist and match docs.

Acceptance

Fresh clone → pip install -e .[dev] → pytest passes in CI.

Lint/format passes in CI.

Phase 1 — “Single command demo” + deterministic artifacts

Goal: Recruiter can run one command and get a clean artifact bundle.

Tasks

Add python -m ns8lab.cli demo that:

generates small datasets for A/B/C (fast)

trains 1 baseline per task (fast)

writes a single reports/results/latest/ snapshot:

summary.json / summary.csv

key plots copied/linked (confusion matrix for A/B, parity plot for C, leakage plot for B)

Introduce “artifact contract validator”:

script scripts/validate_artifacts.py that checks each run folder contains required files.

Ensure deterministic seeds: seed flows everywhere.

Acceptance

python -m ns8lab.cli demo completes in a few minutes and produces:

reports/results/latest/summary.csv

a small set of plots under reports/results/latest/figures/

Phase 2 — Auto-generated “Results Snapshot” (README stays true)

Goal: README/UI show real results without manual edits.

Tasks

Create ns8lab/reporting.py to:

scan reports/experiments/* and/or runs/*

compute “best by task” based on metric rules (A/B: macro F1 then acc; C: RMSE then R²)

export:

reports/results/index.csv

reports/results/index.json

reports/results/latest.md (a markdown table you can embed/link)

Update Streamlit “Results Browser” to read only from reports/results/index.* (one canonical source).

Optional: add a make snapshot target.

Acceptance

Running make snapshot updates results tables + UI reflects it immediately.

README links to reports/results/latest.md instead of hardcoding metrics.

Phase 3 — Task B “leakage honesty” visuals (make it obvious)

Goal: A recruiter instantly understands why Task B hits 1.0 in one split and collapses in another.

Tasks

Add standard plots for Task B:

Split Comparison Bar Chart

x: model

bars: group_mode=n vs group_mode=nk (or “holdout-k”)

y: macro F1 / accuracy

Confusion matrix for both split modes (side-by-side)

Leakage diagnostic plot

show distribution of k_bucket across train/test for each grouping strategy

if using GroupKFold: show which groups are held out

UI additions:

“Task B: Leakage Lab” tab

a single selector for “split mode”

explanation callouts: what is grouped, what is held out, what failure mode is being tested

Acceptance

UI page can toggle split mode and clearly shows performance + why.

Phase 4 — Task C “scale sanity check” visuals (make it obvious)

Goal: Show that features encode scale (N) and what breaks if you stress generalization.

Tasks

Add Task C plots:

Predicted vs True N scatter (parity line)

Residuals vs True N

Generalization stress test

train on N in range A, test on unseen N in range B

plot performance vs “distance” from training range

UI additions:

“Task C: Scale Regression” tab

metric cards: RMSE/R²

slider to choose “train N range / test N range” (read-only mode can show precomputed runs)

Acceptance

UI shows parity + residuals + “unseen N” stress result.

Phase 5 — Dataset Explorer upgrades (turn your CSVs into a story)

Goal: Make the datasets feel tangible: “this is what the model sees.”

Tasks

Dataset Explorer:

Load task_a.csv, task_b.csv, task_c.csv if present

Show:

class balance (A/B)

N and k distributions

feature distributions (entropy, symmetry, FFT ratios)

Add 2D projection (PCA) colored by target to show separability.

Add “render grid sample” (already conceptually in your UI notes) but label it explicitly as sanity-check only (not model input).

Acceptance

A recruiter can click through A/B/C datasets and immediately see patterns + distributions.

Phase 6 — Packaging + “how to use in interviews”

Goal: Make it easy to talk through: architecture, decisions, tradeoffs.

Tasks

Add docs/INTERVIEW_GUIDE.md:

60-second pitch

“why synthetic data / failure modes”

leakage story (Task B)

invariance story (Task A)

scaling story (Task C)

Add docs/ARCHITECTURE.md:

data flow diagram (generate → features → pipeline → tuning → artifacts → UI)

Optional: a short recorded GIF/video in README (UI browsing results)

Acceptance

docs/INTERVIEW_GUIDE.md exists and maps to UI tabs + artifacts.

Phase 7 — Optional: Deploy Streamlit UI (shareable demo)

Goal: A public link recruiters can click.

Tasks

Add Streamlit deployment notes:

read-only mode by default

demo artifact bundle committed under reports/results/latest/ (small)

Ensure app runs without training (no heavy compute).

Acceptance

UI runs fully from precomputed artifacts only.