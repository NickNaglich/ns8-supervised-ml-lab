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
