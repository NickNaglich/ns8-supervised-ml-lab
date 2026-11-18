# Benchmark contract

**Application target: not admitted.** This repository is a synthetic benchmark
for engineered-feature invariance, grouping/leakage and scale encoding, and a
reproducible ML-engineering showcase. [Machine contract](../benchmark-contract.json)
pins this definition to the unchanged source and saved witness.

The documented repo-local kernel is `G[i,j] = 1 + ((i + k*j) % N)`, with zero-based
indices `0..N-1` and displayed values `1..N`. **k is a slope**, not a phase.
Views apply vertical flip for first-letter B, horizontal flip for second-letter R,
then transpose for third-letter B; order is TLF,TRF,BLF,BRF,TLB,BLB,BRB,TRB.
Historical authorial intent is unqualified; this records actual source behavior.
LAB instead uses product residues plus additive phase. ToneSight's fixed8 is
another input/domain contract. Neither is an admitted bridge to these eight view
labels or this feature interface; no shared name/number establishes transfer.

At `hist_bins=16`, X is the ordered **48 `feat_*` columns**: histogram/entropy/gini,
row/column/value statistics, lag correlations, symmetry and FFT energy. Raw grids,
parameter/target/group metadata and pipeline identity are outside X. Native
extraction stays the default; explicit C-order extraction keeps separate identity.
Metadata exclusion does not hide labels automatically: featuremax already gives N.
Other bin counts change the schema; the 48-column profile is explicitly bounded.

TaskA labels are selected generation-view provenance. At k=1, TLF and its
transpose TLB can be the same logical grid; saved rows0/4 also have identical48
native feature inputs. Keep both labels and ambiguity: no unique origin inverse
is promised. [Target decision](APPLICATION_TARGET_DECISION.md) and
[saved witness](../examples/task_a_origin_witness.json) retain the existing proof.
TaskB predicts k modulo the bucket count (default8), a synthetic grouping/leakage
diagnostic. TaskC predicts N, already solved exactly by `feat_value_max`.

Grouping **Nk means `(N,k%N)`**, with N allowed on both sides; it is not unseen-N
generalization. N grouping asks only finite heldout-size transfer. Row splitting
does not claim grouped transfer. Existing outer/inner class support and group
checks, train/fold-only preprocessing and no-fallback behavior remain unchanged.
The fixed fresh-control profile is N16/24/32,k1..8,all8views,192rows, seed0/.25
Nk holdout (18/6groups,144/48rows). That is a named profile, not every API default.

Use exact equality/ambiguity and train-only majority as simple references; TaskB
also has the matched standardized1NN baseline. Its prior N32 holdout retained
18.75% logistic versus31.25%1NN. The fixed TaskA control retained12.5% native and
canonical accuracy, eight changed predictions and no metric gain. Historical
scores/labels stay separate. Whole-table exact ceilings are not heldout scores;
layout/roundoff can change bit groups. No additional training is implied here.

A real application needs a named producer, observed schema/units, independently
measured target, receiving action/error or acquisition costs, native baseline,
source/time/group/duplicate structure and data rights. Until supplied, keep this
explicit benchmark interpretation. This is a scientific input/claim contract,
not a new workflow or approval framework. No generator/function/default/data,
model, score, executable or fixture is changed by this documentation.
