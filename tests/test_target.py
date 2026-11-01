
import pandas as pd
import pytest

from careflow.target import build_admission_target


def test_target_excludes_no_disposition():
    df = pd.DataFrame({
        "ADMITHOS": [1, 0, 0],
        "NODISP": [0, 0, 1],
    })

    result = build_admission_target(df)

    assert len(result) == 2
    assert result["admission_target"].tolist() == [1, 0]


def test_missing_required_column():
    df = pd.DataFrame({"ADMITHOS": [1, 0]})

    with pytest.raises(ValueError):
        build_admission_target(df)
