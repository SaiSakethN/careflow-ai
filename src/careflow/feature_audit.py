
from pathlib import Path

import pandas as pd

from careflow.ingestion import load_ed_data
from careflow.target import build_admission_target


CANDIDATES = [
    "AGE",
    "SEX",
    "ARRTIME",
    "VMONTH",
    "VDAYR",
    "RESIDNCE",
    "CATRIAGE",
    "TEMPF",
    "PULSE",
    "RESPR",
    "BPSYS",
    "BPDIAS",
    "POPCT",
    "PAINSCALE",
    "WAITTIME",
    "LOV",
]


def main():
    root = Path(__file__).resolve().parents[2]
    df = load_ed_data(
        root / "data" / "raw" / "ed2022_sas.sas7bdat"
    )
    df = build_admission_target(df)

    print(f"Eligible visits: {len(df):,}\n")

    for column in CANDIDATES:
        if column not in df.columns:
            print(f"{column:12} NOT FOUND")
            continue

        values = df[column]

        missing = int(values.isna().sum())

        if pd.api.types.is_numeric_dtype(values):
            coded_missing = int(values.isin([-7, -8, -9]).sum())
        else:
            coded_missing = int(
                values.isin([b"-7", b"-8", b"-9"]).sum()
            )

        print(
            f"{column:12} "
            f"dtype={str(values.dtype):9} "
            f"NaN={missing:5} "
            f"coded_missing={coded_missing:5} "
            f"unique={values.nunique()}"
        )


if __name__ == "__main__":
    main()
