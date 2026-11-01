
import pandas as pd
import pytest

from careflow.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_features,
)


def test_missing_codes_and_arrival_time():
    data = {
        column: [1, 2]
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    }
    data["PAINSCALE"] = [-9, 0]
    data["ARRTIME"] = [b"0930", b"-9"]

    result = build_features(pd.DataFrame(data))

    assert pd.isna(result.loc[0, "PAINSCALE"])
    assert result.loc[1, "PAINSCALE"] == 0
    assert result.loc[0, "ARRIVAL_HOUR"] == 9
    assert pd.isna(result.loc[1, "ARRIVAL_HOUR"])


def test_missing_required_column():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"AGE": [25]}))
