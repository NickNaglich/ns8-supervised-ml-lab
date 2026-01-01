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
