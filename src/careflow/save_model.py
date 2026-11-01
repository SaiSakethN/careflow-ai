
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

import sys
from importlib.metadata import version

from careflow.features import FEATURE_COLUMNS, build_features
from careflow.ingestion import load_ed_data
from careflow.split import split_by_hospital
from careflow.target import build_admission_target
from careflow.train_xgboost import build_xgboost_model


THRESHOLD = 0.20


def main():
    root = Path(__file__).resolve().parents[2]

    dataset_path = root / "data" / "raw" / "ed2022_sas.sas7bdat"
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = build_admission_target(load_ed_data(dataset_path))

    train_df, validation_df, test_df = split_by_hospital(df)

    X_train = build_features(train_df)
    y_train = train_df["admission_target"]

    model = build_xgboost_model()
    model.fit(X_train, y_train)

    model_path = artifact_dir / "admission_xgboost.joblib"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": "careflow-admission-xgboost",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "training_visits": len(train_df),
        "validation_visits": len(validation_df),
        "test_visits": len(test_df),
        "training_hospitals": train_df["HOSPCODE"].nunique(),
        "research_only": True,
        "dependencies": {
            "python": sys.version.split()[0],
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "joblib": version("joblib"),
            "shap": version("shap"),
        },
    }

    metadata_path = artifact_dir / "model_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Training visits: {len(train_df):,}")
    print(f"Decision threshold: {THRESHOLD}")


if __name__ == "__main__":
    main()
