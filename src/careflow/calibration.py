
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from careflow.ingestion import load_ed_data
from careflow.target import build_admission_target
from careflow.features import build_features
from careflow.split import split_by_hospital
from careflow.train_xgboost import build_xgboost_model


def main():
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "raw" / "ed2022_sas.sas7bdat"

    df = build_admission_target(load_ed_data(data_path))
    train, validation, _ = split_by_hospital(df)

    X_train = build_features(train)
    X_val = build_features(validation)

    y_train = train["admission_target"]
    y_val = validation["admission_target"]

    model = build_xgboost_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    print("XGBoost validation calibration")
    print(f"Brier score: {brier_score_loss(y_val, probabilities):.4f}")
    print(f"Log loss:    {log_loss(y_val, probabilities):.4f}")
    print(f"Mean predicted probability: {probabilities.mean():.4f}")
    print(f"Actual admission rate:      {y_val.mean():.4f}")

    observed, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    print("\nPredicted | Observed")
    print("-" * 22)

    for pred, obs in zip(predicted, observed):
        print(f"{pred:9.4f} | {obs:.4f}")


if __name__ == "__main__":
    main()
