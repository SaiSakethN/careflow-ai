
from pathlib import Path

import pandas as pd

from careflow.ingestion import load_ed_data


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / "ed2022_sas.sas7bdat"
    df = load_ed_data(path)

    print("\nAdmission vs observation:")
    print(pd.crosstab(df["ADMITHOS"], df["OBSHOS"]))

    columns = [
        "ADMITHOS",
        "OBSHOS",
        "NODISP",
        "OBSDIS",
        "OTHDISP",
        "ADMIT",
        "ADISP",
    ]

    for column in columns:
        if column not in df.columns:
            continue

        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(20))

    print("\nAdmission flag combinations:")
    print(
        df.groupby(["ADMITHOS", "OBSHOS"], dropna=False)
        .size()
        .reset_index(name="visits")
    )


if __name__ == "__main__":
    main()
