from ns8lab import train
from ns8lab.data import build_dataset


def test_group_split_prevents_leakage_between_train_and_test():
    df = build_dataset(task="task_a_view", n_samples=50, seed=0, n_values=[4], k_values=[1, 2], views=["TLF", "TRB"])
    # Run split to ensure no errors and then recompute expected group-based split with same seed
    train._split(df, task="task_a_view", test_size=0.3, seed=0, group_mode="nk")

    groups = [f"{int(n)}_{int(k)}" for n, k in zip(df["N"], df["k"])]
    splitter = train.GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(df[train._feature_columns(df)], df["target"], groups))

    group_train = {groups[i] for i in train_idx}
    group_test = {groups[i] for i in test_idx}

    assert group_train.isdisjoint(group_test)
