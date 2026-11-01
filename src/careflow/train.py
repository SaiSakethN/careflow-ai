
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from careflow.ingestion import load_ed_data
from careflow.target import build_admission_target
from careflow.features import (
    build_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from careflow.split import split_by_hospital


def build_model():
    numeric_columns = NUMERIC_FEATURES + ["ARRIVAL_HOUR"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate(model, X, y, name):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name} metrics")
    print(f"PR-AUC:    {average_precision_score(y, probabilities):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, probabilities):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y, predictions, zero_division=0):.4f}")



def evaluate_thresholds(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]

    print("\nValidation threshold comparison")
    print("Threshold | Precision | Recall | F1")
    print("-" * 40)

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y, predictions, zero_division=0
        )
        recall = recall_score(
            y, predictions, zero_division=0
        )
        f1 = f1_score(
            y, predictions, zero_division=0
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:.4f}"
        )


def main():
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "raw" / "ed2022_sas.sas7bdat"

    df = build_admission_target(load_ed_data(raw_path))
    train, validation, test = split_by_hospital(df)

    X_train = build_features(train)
    X_validation = build_features(validation)

    y_train = train["admission_target"]
    y_validation = validation["admission_target"]

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, "Training")
    evaluate(model, X_validation, y_validation, "Validation")

    evaluate_thresholds(model, X_validation, y_validation)

    print(f"\nHeld-out test visits: {len(test):,}")
    print("Test set reserved for final evaluation.")


if __name__ == "__main__":
    main()
