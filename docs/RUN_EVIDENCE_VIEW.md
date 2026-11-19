# Recorded run evidence

Run Details now exposes `split.json` and optional `baseline.json` alongside
metrics. It shows recorded outer train/test rows and group counts, inner-fold
counts, ordered feature count, and a baseline's name, saved metrics, prediction
count and scope. Native classification runs normally have no baseline record;
absence is shown, never synthesized. Values are read, not recomputed.

The reader opens selected JSON/config/signature files only. It does not open a
model, fit/predict, generate a grid, extract features, rewrite a run or create
missing identities. Each new evidence JSON/signature/config read has a 2 MiB
limit. Missing, malformed, oversized and unsupported split records are reported
explicitly. Schema 1 requires nonempty aligned row positions, sample IDs and
optional groups, unique positions, disjoint outer/inner groups and rows, and
inner folds that partition the recorded outer training rows consistently.
These structural checks do not establish class support, authentication or that
the recorded computation actually occurred.

## Comparison states

The comparison reports separate facts for task, reported dataset identity,
ordered feature schema, grouping, outer training and test membership. Membership
includes recorded row positions, sample IDs and groups, independent of list
ordering. Sample IDs or matching configuration alone never establish data
identity. The current native train/tune paths do not persist a complete dataset
signature, so their records will normally remain **unverified**.

- **incompatible**: at least one available required fact differs. This takes
  precedence over other missing facts.
- **unverified**: no observed mismatch, but at least one required fact is missing
  or invalid. Conflicting signatures within a run also remain unverified.
- **matched_recorded_evidence**: all listed facts are present and equal. Even
  this is only agreement between reports, not authenticated provenance,
  identical model/execution identity or evidence of improved performance.

Raw metrics remain visible side by side as descriptive recorded values. The UI
no longer presents a gain-style metric delta. It never retroactively assigns
split/data identity to historical runs. Dataset identity uses the existing
reported metrics/sidecar signature conventions; it does not certify that a
reported string is the hash of the actual observed input or includes all source,
runtime and feature-pipeline details. The feature is evidence navigation and
bounded comparison, not a model benefit or generalization result.

## Focused verification

`python -m pytest tests/test_run_evidence_view.py` uses independent hand-counted
JSON fixtures: normal and row-grouped splits; missing legacy records; different
tasks/data/schemas/memberships; contradictory signatures; malformed/nonfinite/
oversized files; and a fake Streamlit surface executing the actual details and
comparison functions. It forbids gain-style metric rendering and checks that
reader imports do not load sklearn, training code or joblib. No model, feature
extraction, native Streamlit server or browser is required. YAML is imported
only when its optional path is used; an unavailable YAML parser is reported as
invalid/unavailable evidence, never treated as a successful parse.

A test fixture is not a user run or performance measurement. If an owner has no
existing run folder available, usefulness on that real artifact remains
unverified; no new training is needed merely to populate this view.
