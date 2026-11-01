
from pathlib import Path

import pandas as pd

from careflow.ingestion import load_ed_data


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / "ed2022_sas.sas7bdat"
    df = load_ed_data(path)

    variables = [
        "ADMITHOS",
        "OBSHOS",
        "AGE",
        "SEX",
        "ARRTIME",
        "WAITTIME",
        "LOV",
    ]

    for name in variables:
        print(f"\n{'=' * 50}")
        print(f"VARIABLE: {name}")

        if name not in df.columns:
            print("Not found")
            continue

        series = df[name]

        print(f"Data type: {series.dtype}")
        print(f"Missing: {series.isna().sum():,}")
        print("Top values:")
        print(series.value_counts(dropna=False).head(12))

    print("\nOther potentially relevant columns:")
    keywords = ["DISP", "ADMIT", "OBS", "TRIAGE", "VITAL"]

    matches = [
        col for col in df.columns
        if any(word in col.upper() for word in keywords)
    ]

    print(matches)


if __name__ == "__main__":
    main()
