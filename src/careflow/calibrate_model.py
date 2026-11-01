
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from careflow.features import build_features
from careflow.ingestion import load_ed_data
from careflow.split import split_by_hospital
from careflow.target import build_admission_target
from careflow.train_xgboost import build_xgboost_model


def report(name, y, probabilities):
    print(f"\n{name}")
    print(f"PR-AUC: {average_precision_score(y, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, probabilities):.4f}")
    print(f"Brier: {brier_score_loss(y, probabilities):.4f}")
    print(f"Log loss: {log_loss(y, probabilities):.4f}")
    print(f"Mean risk: {probabilities.mean():.4f}")


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / "ed2022_sas.sas7bdat"

    df = build_admission_target(load_ed_data(path))
    train, validation, test = split_by_hospital(df)

    X_train = build_features(train)
    X_val = build_features(validation)

    y_train = train["admission_target"]
    y_val = validation["admission_target"]

    # Original XGBoost model
    baseline = build_xgboost_model()
    baseline.fit(X_train, y_train)

    # Split training hospitals into three non-overlapping groups
    group_cv = list(
        GroupKFold(n_splits=3).split(
            X_train,
            y_train,
            groups=train["HOSPCODE"],
        )
    )

    # Train calibrated XGBoost using grouped folds
    calibrated = CalibratedClassifierCV(
        estimator=build_xgboost_model(),
        method="sigmoid",
        cv=group_cv,
        ensemble=True,
    )

    calibrated.fit(X_train, y_train)

    # Compare both models on validation data
    report(
        "Original XGBoost",
        y_val,
        baseline.predict_proba(X_val)[:, 1],
    )

    report(
        "Calibrated XGBoost",
        y_val,
        calibrated.predict_proba(X_val)[:, 1],
    )

    print(f"\nUntouched test visits: {len(test):,}")


if __name__ == "__main__":
    main()
