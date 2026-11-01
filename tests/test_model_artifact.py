
import json
from pathlib import Path

import joblib
import numpy as np

from careflow.features import FEATURE_COLUMNS, build_features
from careflow.ingestion import load_ed_data
from careflow.split import split_by_hospital
from careflow.target import build_admission_target
from careflow.train_xgboost import build_xgboost_model


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def test_saved_model_matches_retrained_model():
    model_path = ARTIFACT_DIR / "admission_xgboost.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    assert model_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["features"] == FEATURE_COLUMNS
    assert metadata["threshold"] == 0.20

    df = build_admission_target(
        load_ed_data(ROOT / "data/raw/ed2022_sas.sas7bdat")
    )

    train_df, validation_df, _ = split_by_hospital(df)

    X_train = build_features(train_df)
    y_train = train_df["admission_target"]

    X_validation = build_features(validation_df)

    retrained_model = build_xgboost_model()
    retrained_model.fit(X_train, y_train)

    saved_model = joblib.load(model_path)

    expected = retrained_model.predict_proba(
        X_validation
    )[:, 1]

    actual = saved_model.predict_proba(
        X_validation
    )[:, 1]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-7,
    )
