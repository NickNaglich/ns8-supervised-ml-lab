# Task A at its existing engineered-feature interface

This opt-in no-fit audit uses unchanged `generate_grid` and canonical
`extract_features(hist_bins=16)`. Task A predicts its native view label from
48 source-ordered engineered features, never raw arrays or N/k/view metadata.
All eight actual source views are retained: TLF,TRF,BLF,BRF,TLB,BLB,BRB,TRB.

Freeze before actual feature/collision outcomes: full factorial N16/24/32,k1..8,
eight views =192rows/118784rawcells, at most200000rawcells/20000rowpairs. The
explicit48column schema, exact source/runtime/protocol and budgets are saved
before extracting the table. `freeze` then a single `run` saves all rows, X,
exact feature keys, memberships, label counts, all majority ties and witnesses.
Output belongs outside source at a new task evidence path. No fits, split/model
search, changed representation, default or label, or historical rerun.

Exact input states are ordered tuples of binary64 `float.hex()` strings across
every feature, including the sign of zero. No rounding or tolerance participates
in these groups; hashes are not equality evidence. Signed-zero distinctions can
make this a conservative bit-exact bound rather than a numerical-invariance
bound. Pairwise atol1e-12,rtol0 matches are a separate diagnostic list: approximate
matching is not necessarily transitive, yields no hard equivalence groups and
supports no deterministic accuracy ceiling here.

For the fixed finite table, every deterministic function of exactly X must pick
one label for all rows of a single equal-X group g. It can get at most the largest
label count m_g correct in that group. Groups partition the table, so total
correct <=sum_g m_g. Choosing any maximizer separately for each distinct input
attains this count (a whole-table oracle lookup, not a trained/heldout model).
Thus the empirical deterministic maximum is exactly sum_g m_g /192. We retain
all maximizing ties, not a selected winner. This is not a population/future-data
limit, model performance, statistical significance or a change to Task A labels.

Raw arrays serve only independent alias controls. Raw equality uses shape and
exact logical row-major int64 values, not metadata/hash alone. Native C/F layout
is preserved during extraction and recorded afterward. Identical logical raw
arrays might yield bit-different features through layout/reduction roundoff;
report such splits, never force the feature partition to coarsen raw groups.
Distinguish conflicting labels on identical raw grids (intrinsic view-origin
aliases by design) from exact-X conflicts between different raw grids (additional
representation collisions). Their independent empirical bounds need not nest if
the implementation leaks layout/roundoff distinctions. No forced contiguous
copy, re-extraction, alternate tolerance or repaired extractor is selected.

Prior Task B diagnostics motivated this separate Task A audit. This source uses
the canonical extractor/generator directly; no Task B fitting-runner dependency
is included. Historical Task A .59/.44 remain reported outcomes with
their original domain/provenance limitations; this audit does not reproduce or
invalidate them, label Task A broken or establish external benefit.

Runner: experiments/task_a_identifiability.py freeze|run --out TASK_EVIDENCE.
Run with the admitted environment and source/root plus source/src explicitly on
the import path, outside source cwd, bytecode/cache disabled. `freeze` does not
extract a grid/feature row. `run` verifies frozen inputs/source and refuses a
second STARTED marker or RESULT. New focused tests use only hand-solved small
helper fixtures, not the actual192row table.


## Saved-input reproduction boundary

This repository contains source, tests and scoped summaries, not the original
192-row files, model binaries or local receipts. The experimental runners guard
specific saved-input hashes; they stop on mismatch rather than adapt the trial.
An exact replay requires compatible admitted inputs/runtime. Fresh source runs
have fresh provenance and must not be represented as the earlier frozen receipt.
Generated runtime/provenance outputs belong outside the source checkout.

## Previously reviewed finite result

For the recorded NumPy2.2.6 cohort, exact native X had152groups and empirical
oracle maximum19/24; logical raw-array maximum15/16. One logical raw alias
(N24,k1,BLF/TRB) had different FFT ratio bits with native C/F layout. These are
whole-table input-identifiability diagnostics, not fitted/heldout accuracy or
population limits. No audit was rerun for this source integration.
