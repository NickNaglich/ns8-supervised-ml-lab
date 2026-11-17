# Aggregated Model Cards (Best-So-Far)

## Task A - View Classification
- Recommended config: `configs/task_a.yaml`
- Grouping: (N,k)
- Best model: LogisticRegression
- Metrics: accuracy 0.59, macro F1 0.51 (seed 0, run 20251231_232904)
- Notes: Seed 1 remains lower (acc 0.44/F1 0.40, run 20251231_232942). Group-by-N trial also underperformed. Further stability likely needs more data/feature tweaks.

## Task B - k-bucket Classification
- Recommended config: `configs/task_b_groupN.yaml` (grouping by N only)
- Best models: LogisticRegression / SVM RBF / RandomForestClassifier (tied)
- Metrics: accuracy 1.00, macro F1 1.00 (seeds 0 and 1)
- Notes: Grouping by (N,k) can zero-out performance; stick to group-by-N for this task.

## Task C - Regression (predict N)
- Recommended config: `configs/task_c.yaml` (seed 0) or `configs/task_c_seed1.yaml` (seed 1)
- Best model: RandomForestRegressor
- Metrics: RMSE ~0.0, R² ~1.0 (seeds 0 and 1)
- Notes: Ridge also performs well (RMSE ~0.025–0.03). RF remains strongest.

## Grouped-validation correction (2026-11-16)

The scores above are historical artifacts, not results of the corrected source.
The old Task B N-only recommendation used just N=32 and could not make a
disjoint group holdout with the current split implementation. Its successful
and failed summaries are preserved; their historical split provenance is not
established by the small checked-in summaries. Holding out (N,k) at a single
N with k=1..8 also holds out entire k-bucket classes; failure is informative.

Task B now explicitly defaults to sizes 16,24,32 with N grouping. Recommended
Task B configs declare this changed domain. It is a different finite transfer
question, not a repair of old scores. Both train/tune retain outer and inner
memberships in split.json, use group-aware inner CV, and reject infeasible
groups/class support before any fit. nk groups use k modulo N. Explicit none
means row splitting and does not claim group generalization. Remaining spatial
or feature equivalences need a separate audit for any experiment domain.

Task C writes a matched-input no-fit baseline in baseline.json:
prediction=feat_value_max=N for this synthetic complete-square generator.
Perfect regression alone adds no demonstrated value over that identity.
Raw grids are excluded from model inputs; retained features can expose labels.
See docs/GROUPED_VALIDATION.md for the corrected contract and bounded protocol.
