# One controlled Task A classifier comparison

No production training/default changes. Reuse the accepted192-row native48X
and raw/labels (N16/24/32,k1..8,all8views); no native192 replay. Freeze exact
input/source/runtime/schema and row membership before outcomes. Reuse existing
feature selector/build_groups(nk), same GroupShuffleSplit(seed0,test_size.25)
outer semantics; no unused inner CV. Require24groups,18/6groups,144/48rows,
all8classes and zero identical logical-raw origin crossing; mismatch stops.

Use actual current feature_variants.c_order_features exactly192times. Compare
all48float.hex in frozen source order against the prior saved experimental
C192 table. Mismatch stops before fits. Equal numbers do not relabel old data:
new extraction uses current pipeline_metadata; legacy native provenance remains
an artifact reference. Metadata stays outside model X. Both tabular X matrices
are float64 C-order with unchanged values and common train/test indices.

Exactly two StandardScaler+LogisticRegression fits, classifier C1,lbfgs,
max_iter1000,tol1e-4,random_state0. Scalers fit only144training rows. No search,
CV evaluation, alternate seed/model, refit/retry after outcomes. Save all8-label
confusions, per-row predictions/exact max-decision ties, warnings/iterations and
descriptive time. Model bundles separately include feature identity, source/data
pins, ordered schema, frozen split and hyperparameters. No old model overwrite.

120s external process timeout retains partial JSONL/events;16MiB output limit;
256MiB measured own-process peak target (not an OS-enforced quota). Import-time
preflight measures the existing runtime. Stop and retain partials on a limit.
Single-thread environment, existing runtime read-only; no installs/provider.

This comparison concerns one fixed Nk holdout, not unseen-size generalization,
significance/population accuracy or learned/application benefit. Worse/equal
scores and intrinsic view-origin aliases remain. Prior19/24 and151/192 are
whole-table oracle bounds, not fitted scores for this new partition.

Runner: experiments/canonical_model_control.py freeze|run --audit OLD_AUDIT
--prior-corder OLD_VARIANT_FEATURES --out NEW_OUTPUT. Explicit selected source
paths/admitted runtime, outside source cwd, bytecode disabled. Only new fake-
control tests run; they consume zero real fits/extractions. No successor.


## Saved-input reproduction boundary

This repository contains source, tests and scoped summaries, not the original
192-row files, model binaries or local receipts. The experimental runners guard
specific saved-input hashes; they stop on mismatch rather than adapt the trial.
An exact replay requires compatible admitted inputs/runtime. Fresh source runs
have fresh provenance and must not be represented as the earlier frozen receipt.
Generated runtime/provenance outputs belong outside the source checkout.

## Previously reviewed negative result

On the fixed192-row Task A cohort, GroupShuffleSplit(seed0,test_size0.25) over
(N,k) gave144training and48test rows. Current canonical192X matched the saved
C-order reference before two LogisticRegression(C1,lbfgs,max_iter1000,tol1e-4,
random_state0) fits, one per representation with train-only StandardScaler.
Both native and canonical scored12.5% accuracy and macroF1=0.11538461538461539;
deltas were zero. Confusion matrices matched, but8of48 predictions differed.
There were no fit/convergence warnings (21/22iterations). This result is retained
without model/seed retries. It does not establish predictive improvement,
significance, population or unseen-size generalization, nor invalidate the
historical.59/.44 scores from other domains/provenance. No comparison was rerun
for integration. Original models, data and receipts remain outside Git.
