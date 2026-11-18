# Explicit C-order dataset option

Native remains the default. `extract_features`, formulas, histogram semantics,
labels, train/tune and native cache/export behavior are unchanged. Native rows
retain their old schema; no new pipeline metadata is invented for native data.

```
build_dataset("task_a_view", 2, n_values=[24], k_values=[1],
              views=["BLF", "TRB"], feature_variant="c_order_int64_v1",
              cache_path="new-stable.csv")
python -m ns8lab.cli dataset --task task_a_view --n-samples 2 --n-values 24 \
  --k-values 1 --views BLF TRB --feature-variant c_order_int64_v1 --output new-stable.csv
```

The opt-in copies an already int64 square array to C-order, then invokes the
unchanged extractor with the requested histogram bins. No rounding or dtype
coercion. Stable generation requires positive integer sample count/bins.

Each stable row stores `feature_variant`, `feature_pipeline_id` and canonical
JSON `feature_pipeline_spec`, outside `feat_*` model inputs. The ID hashes the
spec binding variant, actual native/wrapper source SHA256, histogram bins and
declared Python/NumPy/platform runtime fields. Changed bins/source/runtime give
a distinct identity. This is not a cross-platform bit-stability certification
or a model-compatibility proof. Preserve the identity when exporting/consuming.

Stable API caches and CLI exports accept CSV or Parquet (only with an already
installed optional engine). They reject an existing target, including a broken
symlink, before RNG/generation/extraction; exclusive create also refuses a target
appearing later. No implicit directories, overwrite, cache reader or migration.
Serialization failures attempt cleanup only of their newly created partial.
CSV identity strings persist. The recorded exact-bit CSV check used pandas
`float_precision="round_trip"` on pandas3.1.1, which emitted a deprecation warning.
Select a supported exact-roundtrip reader for the consumer/runtime in use; this
option does not add a cache reader or certify arbitrary CSV parser behavior. Native caching behavior stays
as before, including its existing overwrite semantics.

This ID is distinct from prior `experimental_c_order_v1_20261117`; old trial
outputs/models/caches are not relabeled. This option supports dataset generation
and explicit export only. Train/tune still build native data; no adoption or
compatibility enforcement was added there. Do not silently combine feature
variants or apply old models to new data merely because column names agree.
The prior layout/finite Task A audits retain their own runtime/domain bounds;
no new model score, population, external or learned-benefit claim is made.
