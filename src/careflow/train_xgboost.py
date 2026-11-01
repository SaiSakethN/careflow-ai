
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

from careflow.ingestion import load_ed_data
from careflow.target import build_admission_target
from careflow.features import (
    build_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from careflow.split import split_by_hospital


def build_xgboost_model():
    numeric_columns = NUMERIC_FEATURES + ["ARRIVAL_HOUR"]

    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate(model, X, y, name):
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n{name}")
    print(f"PR-AUC:  {average_precision_score(y, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, probabilities):.4f}")

    print("\nThreshold | Precision | Recall | F1")
    print("-" * 40)

    for threshold in [0.15, 0.20, 0.25, 0.30]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

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

    model = build_xgboost_model()
    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, "Training")
    evaluate(model, X_validation, y_validation, "Validation")

    print(f"\nHeld-out test visits: {len(test):,}")
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
