from ns8lab.data import build_dataset
from ns8lab.validation import prepare_split


def test_group_split_prevents_leakage_between_train_and_test():
    df = build_dataset(task="task_a_view", n_samples=160, seed=0,
                       n_values=[4, 8, 12], k_values=[1, 2, 3], views=["TLF", "TRB"])
    split = prepare_split(df, "task_a_view", test_size=0.3, seed=0, group_mode="nk")
    outer = split.provenance["outer"]
    assert set(outer["train"]["groups"]).isdisjoint(outer["test"]["groups"])
    for fold in split.provenance["inner"]:
        assert set(fold["train"]["groups"]).isdisjoint(fold["validation"]["groups"])
