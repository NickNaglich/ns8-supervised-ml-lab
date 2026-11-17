# NS8 Lab - Results Snapshot

Recommended configs and metrics (latest runs):

| Task | Grouping | Seed | Model | Metrics | Config |
| --- | --- | --- | --- | --- | --- |
| Task A (view clf) | (N,k) | 0 | LogisticRegression | acc 0.59, F1 0.51 | configs/task_a.yaml (latest run 20251231_232904) |
| Task B (k-bucket) | N-only | 0 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | configs/task_b_groupN.yaml |
| Task B (k-bucket) | N-only | 1 | logreg/SVM/RF (tie) | acc 1.00, F1 1.00 | configs/task_b_groupN.yaml |
| Task C (regress N) | (N,k) | 0 | RF regressor | RMSE ~0.0, R² ~1.0 | configs/task_c.yaml |
| Task C (regress N) | (N,k) | 1 | RF regressor | RMSE ~0.0, R² ~1.0 | configs/task_c_seed1.yaml |

Notes:
- Task A stability across seeds is limited; seed 1 underperforms even with group-by-N. Consider feature tweaks or more data if higher stability is required.
- Task B grouping by (N,k) can zero-out performance; group by N for sane metrics.
- All runs use group-aware splits by default; adjust via `group_mode` if needed.

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
