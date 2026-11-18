# Fresh-data feature comparison

This separate source-only command needs the repository and declared dependencies,
not any private prior audit tables. From the repository root:

```sh
python -m experiments.fresh_model_control prepare --out data/fresh-control
python -m experiments.fresh_model_control validate --out data/fresh-control
```

The parent directory must exist; the output directory must not exist. Preparation
generates the full factorial N=16/24/32, k=1..8, all eight canonical TaskA views:
192 rows/118784 logical cells. It extracts native and `c_order_int64_v1` features
once each per row, preserving generator layouts for native extraction. Both X
tables contain only the same ordered 48 engineered features (`hist_bins=16`).
All labels, raw grids, groups, source/runtime and pipeline identities stay outside
X. Logical raw aliases and their view labels remain intact; any identical raw
origin crossing the split is rejected before fitting.

The protocol/source/runtime/identity and row/split files are frozen before
extraction. A fixed GroupShuffleSplit on (N,k modulo N), seed=0, test_size=0.25
must produce 18/6 disjoint groups and 144/48 rows, with every view in both sides.
No fallback or resampling is permitted. Saved raw values/hashes, layout flags,
both float.hex tables, alias memberships, integrity pins, extraction counters,
time and peak process memory make the no-fit preparation independently usable.
Validation checks saved structure, hashes, source/runtime and fixed memberships
without generating grids, extracting features or fitting. Pins record integrity,
not origin authentication. The output and source/runtime are a coherent run;
changed source/runtime requires a separately identified preparation.

Fitting is a separate **explicit opt-in** command, never preparation's default:

```sh
python -m experiments.fresh_model_control fit --prepared data/fresh-control --out data/fresh-fit
```

This performs exactly two classifier fits and two train-only StandardScaler fits,
native/canonical on identical saved memberships. Fixed LogisticRegression:
C=1, solver=lbfgs, max_iter=1000, tol=1e-4, random_state=0. No CV/search or model
selection is involved. Each result retains test row IDs, true/predicted labels,
all-eight-label confusion/accuracy/macro-F1, warnings and descriptive time;
identity/source/runtime/split/parameters are frozen separately before outcomes.
No model cache is updated. Fit output must be new. Failures preserve freezes and
partial results; a new output is required for a deliberately selected new run.
The fit command checks the manifest is unchanged across validation, then hashes
the exact bytes loaded once for X, memberships and identities against that
validated manifest. Only those in-memory inputs reach model construction/fitting;
later file changes cannot substitute unchecked fitting inputs.

Budget: preparation 192 native + 192 canonical extractions, zero fits; explicit
fit two pipelines, zero extractions. Each command targets 120 seconds and limits
saved output to 16 MiB; time is a caller-enforced process target, not an internal
kill switch. Peak memory measures the whole Python process, including imports.
The small factorial is fixed, not a general benchmark framework.

Fresh native FFT/roundoff/layout/runtime behavior can differ from past saved X.
There is **no historical byte/score identity guarantee** or expectation of gain.
The earlier exact replay runner `canonical_model_control.py` remains unchanged;
its recorded comparison stays 12.5% accuracy for both variants, eight changed
predictions and no metric gain. New scores, if explicitly requested, describe
only this synthetic grouped partition, not population accuracy or learned value.
Intrinsic raw view-origin aliases cannot be recovered by changing representation.
