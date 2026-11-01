
from unittest import result

import pandas as pd


NUMERIC_FEATURES = [
    "AGE",
    "TEMPF",
    "PULSE",
    "RESPR",
    "BPSYS",
    "BPDIAS",
    "POPCT",
    "PAINSCALE",
]

CATEGORICAL_FEATURES = [
    "SEX",
    "RESIDNCE",
    "IMMEDR",
    "VMONTH",
    "VDAYR",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
    "ARRIVAL_HOUR"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare post-triage candidate features without fitting imputation."""

    missing_columns = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["ARRTIME"]
    ) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {sorted(missing_columns)}"
        )

    result = pd.DataFrame(index=df.index)

    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")
        result[column] = values.replace(
            [-7, -8, -9], float("nan")
        )

     # CDC NHAMCS 2022 vital-sign coding corrections
    result["TEMPF"] = result["TEMPF"] / 10.0
    result["PULSE"] = result["PULSE"].replace(998, float("nan"))
    result["BPDIAS"] = result["BPDIAS"].replace(998, float("nan"))

    # Retain documented 1–5 triage levels for the initial model
    result["IMMEDR"] = result["IMMEDR"].where(
        result["IMMEDR"].isin([1, 2, 3, 4, 5])
    )

    arrival = df["ARRTIME"].apply(
        lambda value: (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
        )
    )

    valid_time = arrival.str.fullmatch(r"\d{4}").fillna(False)
    hours = pd.to_numeric(arrival.str[:2], errors="coerce")
    minutes = pd.to_numeric(arrival.str[2:], errors="coerce")

    result["ARRIVAL_HOUR"] = hours.where(
        valid_time
        & hours.between(0, 23)
        & minutes.between(0, 59)
    )

    return result[FEATURE_COLUMNS]
