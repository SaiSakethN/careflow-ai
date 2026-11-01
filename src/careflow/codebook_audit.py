
from pathlib import Path

from careflow.ingestion import load_ed_data
from careflow.target import build_admission_target


COLUMNS = [
    "AGE",
    "SEX",
    "RESIDNCE",
    "CATRIAGE",
    "VMONTH",
    "VDAYR",
    "TEMPF",
    "PULSE",
    "RESPR",
    "BPSYS",
    "BPDIAS",
    "POPCT",
    "PAINSCALE",
]


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / "ed2022_sas.sas7bdat"

    df = build_admission_target(load_ed_data(path))

    for column in COLUMNS:
        print(f"\n{'=' * 50}")
        print(f"{column}")
        print(f"{'=' * 50}")

        values = df[column]

        print("Minimum:", values.min())
        print("Maximum:", values.max())

        if values.nunique() <= 20:
            print("\nValue distribution:")
            print(values.value_counts(dropna=False).sort_index())
        else:
            print("\nPercentiles:")
            print(values.quantile([0, 0.01, 0.5, 0.99, 1]))


if __name__ == "__main__":
    main()
