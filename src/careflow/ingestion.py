
from pathlib import Path

import pandas as pd


def load_ed_data(file_path: Path) -> pd.DataFrame:
    """Load the NHAMCS emergency department SAS dataset."""

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_sas(file_path, format="sas7bdat")

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "raw" / "ed2022_sas.sas7bdat"

    df = load_ed_data(dataset_path)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print("\nFirst 10 columns:")
    print(df.columns[:10].tolist())
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")
