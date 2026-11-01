
import numpy as np
import pandas as pd

from careflow.train import build_model
from careflow.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_baseline_model_fits_and_predicts():
    rng = np.random.default_rng(42)
    n = 100

    data = {
        column: rng.normal(size=n)
        for column in NUMERIC_FEATURES
    }

    for column in CATEGORICAL_FEATURES:
        data[column] = rng.choice([1, 2, 3], size=n)

    data["ARRIVAL_HOUR"] = rng.integers(0, 24, size=n)

    X = pd.DataFrame(data)
    X.loc[0, "PULSE"] = np.nan
    X.loc[1, "CATRIAGE"] = np.nan

    y = pd.Series([0, 1] * 50)

    model = build_model()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (n, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1)
