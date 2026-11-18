# Application target decision: NO-USEFUL-TARGET-YET

The admitted source supports an honest synthetic ML-engineering showcase, not a
qualified ArckN application predictor. README defines engineered-feature
invariance/leakage/scale diagnostics; data.py labels views, k%8 and N from its
own generator metadata. No producer-owned external measurement, downstream
action/cost, or outcome label is admitted here. This is a boundary of these
sources, not a claim that no useful ArckN task exists elsewhere.

## The operator is additive slope, not LAB multiplicative phase

Actual `src/ns8lab/grids.py` computes `1+((r+k*c)%N)` before view flips and
transpose: k is a **kernel slope**. LAB's product-residue phase operator is
`1+((r*c+(phase-1))%N)`: phase is an **additive translation**. With zero-based
(r,c)=(0,1), N=4 and slope=phase=1, ML gives2 and LAB gives1. The runnable check
prints this explicit convention control; its fixture records the selected
generator blob/SHA. This deliberate repo-local convention is not established as
a bug; neither generator nor historical labels/scores are rewritten.

Even the complex encoding differs: `exp(2*pi*i*(r+k*c)/N)` factors into
`exp(2*pi*i*r/N)*exp(2*pi*i*k*c/N)`, an outer product of nonzero vectors, rank1.
That elementary algebra concerns this encoding, not a new numerical rank study,
the LAB product operator, digital-root projections or quantum/application value.
Phase-period/recovery/QFT results do not transfer just because both use N/k grids.

**One actual receiving decision:** may this ML project's reports/UI describe an
eight-view prediction as uniquely recoverable historical origin? For the saved
witness below, no. Keep view labels as legitimate benchmark metadata, preserve
all historical scores, and state ambiguity; do not invent a replacement class.

## Runnable existing-data witness

From repository root, using only Python's standard library:

```sh
python -I -B -S experiments/observable_target_gate.py --witness examples/task_a_origin_witness.json
```

The example copies ONLY saved TaskA rows0/4 (N16,k1,TLF/TLB), original logical
values/layout records, and all48 native float.hex inputs. Original-file hashes
are recorded. The checker reads this supplied file only, never reference paths,
and writes JSON to stdout. It verifies shape/range/row-major int64 hashes and
exact input equality without generating a grid, extracting features or fitting.
Its provenance refs are metadata, not source authentication or proof of origin.

Simplest baseline: compare the exact supplied input and retain both conflicting
original labels. Here TLF's G[r,c]=1+((r+c)%16) is symmetric, so its transpose TLB
has identical logical values. The saved native48 inputs also agree bit-for-bit.
One deterministic prediction for that same X can match at most one of these
two different targets. The reported1/2 bound is ONLY this two-record witness;
it is not fitted accuracy, a heldout/population limit or a new full-table audit.
Raw arrays are independent controls, never new model columns. Signed zero and
roundoff remain distinct; no tolerance equivalence or layout-bit cleanup occurs.

Other evidence already closes tempting detours: the TaskB N32 holdout gave
logistic18.75%, majority12.5%, standardized1NN31.25%; this is finite transfer
evidence, not application utility. The existing TaskC max-feature baseline is
exact. Canonical-layout TaskA control stayed12.5% for both representations with
eight changed predictions and no gain. Do not repeat these experiments merely
to find a better score; none defines an external receiving decision.

The prior native192-row exact-input19/24 and raw15/16 ceilings describe that
whole saved table only, not fitted/heldout/population accuracy. Layout/roundoff
can split logical raw aliases into different feature bits; these partitions need
not nest. Nk grouping prevents that saved group's train/test crossing, but it
does not make origin observable. No current48-row holdout bound is recalculated.

Primary current source (canonical a0f390a5, tree4d734910):
[README](https://github.com/NickNaglich/ns8-supervised-ml-lab/blob/a0f390a5c3e73deb9579f885a9ae301314fe282f/README.md),
[generator](https://github.com/NickNaglich/ns8-supervised-ml-lab/blob/a0f390a5c3e73deb9579f885a9ae301314fe282f/src/ns8lab/grids.py),
[task definitions](https://github.com/NickNaglich/ns8-supervised-ml-lab/blob/a0f390a5c3e73deb9579f885a9ae301314fe282f/src/ns8lab/data.py).
The fixture's original source-file hashes pin the selected older audit evidence;
historical scores retain their separate domains/provenance.

## At most two receiving-workflow choices

1. **Existing ML lab reporting/admission:** use this witness to qualify claims
   and discuss retained information, split leakage and negative controls. That
   useful outcome needs no further training. Before any new prediction task,
   the receiving owner must supply one observed input schema, a real action and
   its cost, a label measured independently of generator parameters, and its
   group/time acquisition structure. A frozen heldout source/time group would
   then test that specific decision against the strongest native baseline.
2. **Existing NS8 sparse-report workflow:** if a real producer needs missing
   phase-count reconstruction, first admit its exact N/phase/nine-channel count
   schema and acquisition/action cost. Use existing exact sparse recovery and
   abstention as the baseline, not an ML surrogate. Additive slope grids in this
   ML project are not the canonical product-residue transport operator; a
   generic histogram cannot justify that transfer. No such producer task is
   admitted by this note, and exact recovery capability alone proves no benefit.

Stop here: no new cohort, extraction, fit/search, relabeling, modelstore update or
application claim. The original192-row audit, intrinsic aliases, saved partition
bounds and historical successes/failures remain unchanged.
