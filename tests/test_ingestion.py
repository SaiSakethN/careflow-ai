
from pathlib import Path

import pandas as pd
import pytest

from careflow.ingestion import load_ed_data


def test_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.sas7bdat"

    with pytest.raises(FileNotFoundError):
        load_ed_data(missing_file)


def test_load_real_dataset():
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (
        project_root / "data" / "raw" / "ed2022_sas.sas7bdat"
    )

    if not dataset_path.exists():
        pytest.skip("CDC dataset not available locally")

    df = load_ed_data(dataset_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (16025, 913)
    assert "AGE" in df.columns
    assert "ADMITHOS" in df.columns
