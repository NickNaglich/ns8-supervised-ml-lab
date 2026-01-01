import pandas as pd

from ns8lab.data import build_dataset


def test_build_dataset_task_a_is_deterministic_and_has_targets():
    df1 = build_dataset(
        task="task_a_view",
        n_samples=5,
        seed=42,
        n_values=[4],
        k_values=[1, 2],
        views=["TLF", "TRB"],
        hist_bins=4,
    )
    df2 = build_dataset(
        task="task_a_view",
        n_samples=5,
        seed=42,
        n_values=[4],
        k_values=[1, 2],
        views=["TLF", "TRB"],
        hist_bins=4,
    )
    pd.testing.assert_frame_equal(df1, df2)
    assert set(df1["target"]).issubset({"TLF", "TRB"})
    assert df1.filter(like="feat_").shape[1] > 0


def test_build_dataset_task_b_k_bucket_labels():
    df = build_dataset(
        task="task_b_kbucket",
        n_samples=8,
        seed=7,
        n_values=[5],
        k_values=[1, 2, 3, 4],
        views=["TLF"],
        k_buckets=4,
        hist_bins=3,
    )
    assert set(df["target_type"]) == {"classification_k_bucket"}
    assert set(df["target"]).issubset({0, 1, 2, 3})
    assert all(df["N"] == 5)
    assert all(df["view"] == "TLF")
