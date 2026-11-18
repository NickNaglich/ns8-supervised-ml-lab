"""Only one test performs real extraction: 3 calls. Plumbing uses fake callbacks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ns8lab import cli, data, feature_variants as variants
from ns8lab.validation import feature_columns


def fake_features(grid, hist_bins=16):
    return {"feat_fake": float(grid[0, 0]), "feat_bins": float(hist_bins)}


@pytest.fixture
def fake_extraction(monkeypatch):
    monkeypatch.setattr(data, "extract_features", fake_features)
    monkeypatch.setattr(variants.native_module, "extract_features", fake_features)


def kwargs():
    return dict(task="task_a_view", n_samples=2, n_values=[4], k_values=[1], views=["TLF"], seed=2)


def test_real_same_values_layouts_stable_exact_and_unmutated():
    # 3 ACTUAL extractions total for this module; independent raw construction.
    base = np.array([[1 + ((23-r+c) % 24) for c in range(24)] for r in range(24)], dtype=np.int64)
    backing = np.array(base[::-1], copy=True, order="C")
    layouts = [np.array(base, copy=True, order="C"), np.array(base, copy=True, order="F"), backing[::-1]]
    vectors = []
    for grid in layouts:
        before = grid.tobytes(order="C"); strides = grid.strides
        values = variants.c_order_features(grid, 16)
        vectors.append(tuple(float(x).hex() for x in values.values()))
        assert grid.tobytes(order="C") == before and grid.strides == strides
        assert grid.dtype == np.dtype("int64") and len(values) == 48
    assert vectors[0] == vectors[1] == vectors[2]


def test_native_default_schema_and_overwrite_behavior_unchanged(fake_extraction, tmp_path):
    default = data.build_dataset(**kwargs())
    explicit = data.build_dataset(**kwargs(), feature_variant="native")
    pd.testing.assert_frame_equal(default, explicit)
    assert not set(variants.IDENTITY_COLUMNS) & set(default.columns)
    path = tmp_path / "native.csv"; path.write_text("old")
    data.build_dataset(**kwargs(), cache_path=path)
    assert path.read_text().startswith("sample_id,")


def test_stable_metadata_roundtrip_separate_from_model_X(fake_extraction, tmp_path):
    path = tmp_path / "stable.csv"
    df = data.build_dataset(**kwargs(), feature_variant=variants.C_ORDER_VARIANT, cache_path=path)
    assert feature_columns(df) == ["feat_fake", "feat_bins"]
    loaded = pd.read_csv(path)
    for name in variants.IDENTITY_COLUMNS:
        assert loaded[name].tolist() == df[name].tolist()
    spec = json.loads(df.feature_pipeline_spec.iloc[0])
    assert spec["hist_bins"] == 16 and spec["feature_variant"] == variants.C_ORDER_VARIANT
    assert spec["runtime"]["numpy"] == np.__version__
    assert hashlib.sha256(df.feature_pipeline_spec.iloc[0].encode("ascii")).hexdigest() == df.feature_pipeline_id.iloc[0]
    assert spec["native_source_sha256"] == hashlib.sha256(Path(variants.native_module.__file__).read_bytes()).hexdigest()
    assert spec["wrapper_source_sha256"] == hashlib.sha256(Path(variants.__file__).read_bytes()).hexdigest()
    other = data.build_dataset(**kwargs(), feature_variant=variants.C_ORDER_VARIANT, hist_bins=8)
    assert other.feature_pipeline_id.iloc[0] != df.feature_pipeline_id.iloc[0]


@pytest.mark.parametrize("kind", ["invalid_variant", "existing", "bad_suffix", "missing_parent", "no_parquet_engine", "empty_stable"])
def test_invalid_or_existing_target_rejected_before_work(monkeypatch, tmp_path, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before RNG/grid/extraction")
    monkeypatch.setattr(data.np.random, "default_rng", forbidden)
    options = kwargs(); options["feature_variant"] = variants.C_ORDER_VARIANT
    expected = (ValueError, FileExistsError, FileNotFoundError, RuntimeError)
    if kind == "invalid_variant": options["feature_variant"] = "experimental_c_order_v1_20261117"
    elif kind == "existing":
        path = tmp_path / "exists.csv"; path.write_text("keep"); options["cache_path"] = path
    elif kind == "bad_suffix": options["cache_path"] = tmp_path / "bad.json"
    elif kind == "missing_parent": options["cache_path"] = tmp_path / "missing/out.csv"
    elif kind == "no_parquet_engine":
        monkeypatch.setattr(data, "find_spec", lambda name: None); options["cache_path"] = tmp_path / "out.parquet"
    else: options["n_samples"] = 0
    with pytest.raises(expected): data.build_dataset(**options)


def test_late_output_collision_never_overwrites(fake_extraction, monkeypatch, tmp_path):
    path = tmp_path / "race.csv"
    original = data.generate_grid
    def create_competitor(*args, **kwargs):
        path.write_text("competitor")
        return original(*args, **kwargs)
    monkeypatch.setattr(data, "generate_grid", create_competitor)
    with pytest.raises(FileExistsError):
        data.build_dataset(**kwargs(), feature_variant=variants.C_ORDER_VARIANT, cache_path=path)
    assert path.read_text() == "competitor"


def test_failed_exclusive_serialization_cleans_only_new_partial(tmp_path):
    class Broken:
        def to_csv(self, stream, index):
            stream.write("partial"); raise RuntimeError("intentional")
    path = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError): data._save_stable_dataset(Broken(), path)
    assert not path.exists()


def test_cli_routes_stable_export_through_api_and_native_stays_csv(monkeypatch, tmp_path):
    calls = []
    class FakeFrame:
        def __len__(self): return 1
        def to_csv(self, path, index): calls.append(("csv", str(path), index))
    def build(**options): calls.append(("api", options)); return FakeFrame()
    monkeypatch.setattr(cli, "build_dataset", build)
    parser = cli.build_parser()
    path = str(tmp_path / "out.csv")
    stable = parser.parse_args(["dataset", "--task", "task_a_view", "--feature-variant", variants.C_ORDER_VARIANT, "--output", path]); stable.func(stable)
    assert len(calls) == 1 and calls[0][1]["cache_path"] == path
    assert calls[0][1]["feature_variant"] == variants.C_ORDER_VARIANT
    calls.clear()
    native = parser.parse_args(["dataset", "--task", "task_a_view", "--output", path]); native.func(native)
    assert calls[0][1]["cache_path"] is None and calls[1] == ("csv", path, False)
    with pytest.raises(SystemExit): parser.parse_args(["dataset", "--task", "task_a_view", "--feature-variant", "bad"])
    with pytest.raises(SystemExit): parser.parse_args(["train", "--task", "task_a_view", "--feature-variant", variants.C_ORDER_VARIANT])


def test_actual_cli_existing_stable_target_rejects_before_work(monkeypatch, tmp_path):
    path = tmp_path / "already.csv"; path.write_text("keep")
    def forbidden(*args, **kwargs): raise AssertionError("must not generate")
    monkeypatch.setattr(data, "generate_grid", forbidden)
    args = cli.build_parser().parse_args(["dataset", "--task", "task_a_view", "--feature-variant", variants.C_ORDER_VARIANT, "--output", str(path)])
    with pytest.raises(FileExistsError): args.func(args)
    assert path.read_text() == "keep"


@pytest.mark.parametrize("dtype", [np.float64, np.int32, bool])
def test_wrapper_no_dtype_coercion(monkeypatch, dtype):
    def forbidden(*args, **kwargs): raise AssertionError("must not extract")
    monkeypatch.setattr(variants.native_module, "extract_features", forbidden)
    with pytest.raises(ValueError): variants.c_order_features(np.ones((4, 4), dtype=dtype), 16)
