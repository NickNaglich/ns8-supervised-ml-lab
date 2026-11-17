# Grouped validation contract

Both training entry points use the same `prepare_split` contract. Outer groups
stay disjoint. Inner `GroupKFold` receives the retained training groups in
`GridSearchCV.fit`; preflight checks the same folds before any model is fitted.
Every classification partition must contain all task classes; regression
partitions need at least two rows. Unsplittable data fails explicitly. Folds may
be fewer than requested when only fewer groups exist, but never fewer than two.
Explicit `group_mode: none` uses row splitting and ordinary stratified/KFold CV;
it makes no group-transfer claim. `nk` identifies slopes modulo N. This handles
that known generator alias, not every spatial or feature equivalence.

Every run writes `split.json` with feature names, original row positions/sample
IDs, outer groups and inner memberships. Effective grouping/fold count is saved
with config. Scalers remain inside pipelines. Invalid folds are rejected before
search construction and fit errors are raised, not turned into NaN scores.

Task B defaults/recommended configs now use N=16,24,32 with N grouping. This
changes the question to transfer across these finite sizes with shared bucket
classes. Old one-N N-only configs are invalid; one-N nk holdouts may withhold
entire classes. Historical zero and perfect summaries remain unchanged and are
not measurements of this correction. Their detailed historical provenance is
not reconstructed here. Task C's `baseline.json` uses the same test feature
matrix: prediction=feat_value_max. That equals N for this synthetic generator,
without a fitted model. Existing learned scores do not establish added value.

## One selected finite comparison, declared before generation/fits

Enumerate exactly N=16,24,32; k=1..8; views=TLF,TRB,BLF,BRF (96 rows), using
existing data/grid/feature functions with 16 histogram bins. Hold N32 out; inner
leave-one-N-out on 16/24. All eight labels must occur in every train/validation
partition. Do not resample, change the split, remove a label or revise features
to improve the outcome.

Before fitting, persist protocol, source hashes, ordered feature table/schema,
all memberships and fingerprints. Exact grids are hashed as C-order little-
endian int64 including shape; feature exact hashes use ordered float.hex strings.
Feature-equivalence audit uses pairwise absolute difference <=1e-12 in EVERY
feature (rtol=0); retain all matching pairs, rather than assuming transitivity.
Also save fingerprints rounded to 12 decimal places, normalizing signed zero;
these are indexing aids, not substitutes for the pairwise audit. Any matching
pair with conflicting labels, any cross-partition exact/feature alias, nonfinite
feature, missing class or group mismatch stops this benchmark BEFORE fit.
Same-label aliases confined to a partition are reported, not silently dropped.

If preflight passes, compare fixed LogisticRegression(C=0.5,max_iter=1000,
random_state=0), train-only majority, and StandardScaler + 1NN on identical X and
memberships. Scalers are fitted separately inside each fold and final training
partition. No search, heldout tuning or extra seeds. Save per-row predictions,
accuracy/macro-F1, and source/data/protocol bindings; failed/no-value outcomes
are acceptable results. This single N32 holdout is not a general scale theorem
or an external ArckN application result. No NS8 recovery/collision theorem is
used: slope grids and phase-translated product counts have different operators.
