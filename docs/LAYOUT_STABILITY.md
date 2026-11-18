# Experimental C-order feature-storage trial

Separate variant `experimental_c_order_v1_20261117`; never replace native defaults
or mix its outputs with old cached CSV/parquet, models or feature artifacts.
Native features have no feature-version token; project0.0.1 is not such a token.
The trial records the native-formula and wrapper-source SHA256 plus variant ID.
Feature names stay the same, so identity must remain outside model X.

Before outcomes, freeze source/schema/runtime and accepted saved raw/X hashes.
Use ONE saved N24/k1 logical matrix from TaskA rows66/71. Construct identical
int64 C/F/negative-row-stride arrays, with reversed backing then reversed view
for the negative-stride case. Verify values/shape/dtype/row-major bytes first.
Extract native48features once per control. Require C/F vectors to match saved
row66/71 and differ; otherwise retain evidence and stop before any wrapper.
This intervenes on storage while holding logical values/dtype fixed; source and
runtime are pinned. It does not isolate a particular internal FFT/reduction
instruction or establish behavior for every platform/layout.

Only if reproduced: one wrapper `np.array(copy=True, order='C')`, with int64 dtype
required, canonical extractor/hist_bins16 unchanged. Once per control; criterion
is all48ordered float.hex values identical and inputs unmutated, not accuracy.
No rounding, atol merging, formula, label, representation dimension or default
change. An explicit C-order copy is the entire experimental intervention.

Only if stable: wrapper once per192 SAVED matrices/118784cells, compare saved
oldX (no native192 replay). Reuse accepted exact-key/group/ceiling logic; retain
all labels/ties and explicit variant identity. Old19/24 describes oldX only;
lower variant ceiling can reflect removal of accidental bit distinctions, not
a regression in the success criterion. Intrinsic raw-origin aliases remain.

Budget3direct-native +3conditional-wrapper controls +192conditional-wrapper
table calls, zero fits/search. Tests use fake extractor callbacks only. Source
generator/defaults, old audit/data/models/caches remain untouched. The whole-
table empirical deterministic bound is not fitted/heldout/population accuracy
or learned/application benefit. No automatic successor or default adoption.

Runner: experiments/layout_stability.py freeze|run --saved-audit OLD_AUDIT
--out NEW_TRIAL. Run from outside source cwd with admitted runtime, selected
source root/src explicit, bytecode/cache disabled. Files are exclusive-created;
a STARTED marker prevents replay. Exact commands/results remain task evidence.


## Saved-input reproduction boundary

This repository contains source, tests and scoped summaries, not the original
192-row files, model binaries or local receipts. The experimental runners guard
specific saved-input hashes; they stop on mismatch rather than adapt the trial.
An exact replay requires compatible admitted inputs/runtime. Fresh source runs
have fresh provenance and must not be represented as the earlier frozen receipt.
Generated runtime/provenance outputs belong outside the source checkout.

## Previously reviewed controlled result

The recorded same-values C/F/negative-stride intervention reproduced the native
C/F difference. Three C-order wrapper controls were exact-bit stable; the saved
192-row trial changed31rows in FFT ratios and yielded151/192 versus native19/24
whole-table oracle maximum. Exact groups both merged and split. This supports
the opt-in storage trial, not a higher-accuracy claim or universal layout result.
The dataset API variant `c_order_int64_v1` has a separate identity from this
`experimental_c_order_v1_20261117` producer. No trial was rerun for integration.
