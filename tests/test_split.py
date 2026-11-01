
import pandas as pd

from careflow.split import split_by_hospital


def test_hospital_groups_do_not_overlap():
    df = pd.DataFrame({
        "HOSPCODE": [i // 5 for i in range(500)],
        "admission_target": [i % 2 for i in range(500)],
    })

    train, validation, test = split_by_hospital(df)

    train_ids = set(train["HOSPCODE"])
    val_ids = set(validation["HOSPCODE"])
    test_ids = set(test["HOSPCODE"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert len(train) + len(validation) + len(test) == len(df)


def test_split_is_reproducible():
    df = pd.DataFrame({
        "HOSPCODE": [i // 5 for i in range(500)],
    })

    first = split_by_hospital(df, random_state=42)
    second = split_by_hospital(df, random_state=42)

    for left, right in zip(first, second):
        assert left.index.tolist() == right.index.tolist()
