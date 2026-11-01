
from pathlib import Path

import pandas as pd

from careflow.ingestion import load_ed_data


def main():
    root = Path(__file__).resolve().parents[2]
    df = load_ed_data(
        root / "data" / "raw" / "ed2022_sas.sas7bdat"
    )

    flags = [
        "NODISP",
        "ADMITHOS",
        "OBSHOS",
        "OBSDIS",
        "OTHDISP",
    ]

    print("Disposition flag combinations:")
    combinations = (
        df.groupby(flags, dropna=False)
        .size()
        .sort_values(ascending=False)
        .rename("visits")
        .reset_index()
    )
    print(combinations.to_string(index=False))

    print("\nNumber of positive disposition flags per visit:")
    print(df[flags].eq(1).sum(axis=1).value_counts().sort_index())

    print("\nCross-tab: admission and no disposition:")
    print(pd.crosstab(df["ADMITHOS"], df["NODISP"]))

    print("\nCross-tab: admission and observation discharge:")
    print(pd.crosstab(df["ADMITHOS"], df["OBSDIS"]))


if __name__ == "__main__":
    main()
