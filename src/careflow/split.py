
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_hospital(
    df: pd.DataFrame,
    random_state: int = 42,
):
    """Split visits without sharing hospitals across partitions."""

    if "HOSPCODE" not in df.columns:
        raise ValueError("HOSPCODE is required")

    if df["HOSPCODE"].isna().any():
        raise ValueError("Hospital identifiers contain missing values")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        splitter.split(df, groups=df["HOSPCODE"])
    )

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state,
    )

    val_idx, test_idx = next(
        second_splitter.split(temp, groups=temp["HOSPCODE"])
    )

    validation = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    train_hospitals = set(train["HOSPCODE"])
    val_hospitals = set(validation["HOSPCODE"])
    test_hospitals = set(test["HOSPCODE"])

    assert train_hospitals.isdisjoint(val_hospitals)
    assert train_hospitals.isdisjoint(test_hospitals)
    assert val_hospitals.isdisjoint(test_hospitals)

    return train, validation, test
