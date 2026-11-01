
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from careflow.features import build_features
from careflow.ingestion import load_ed_data
from careflow.split import split_by_hospital
from careflow.target import build_admission_target
from careflow.train_xgboost import build_xgboost_model


def main():
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "raw" / "ed2022_sas.sas7bdat"
    output_dir = root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_admission_target(load_ed_data(data_path))
    train, validation, _ = split_by_hospital(df)

    X_train = build_features(train)
    X_val = build_features(validation)

    model = build_xgboost_model()
    model.fit(X_train, train["admission_target"])

    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    # Explain the same transformed features used by XGBoost
    X_transformed = preprocessor.transform(X_val)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_explain = pd.DataFrame(
        X_transformed,
        columns=feature_names,
    ).sample(n=min(500, len(X_val)), random_state=42)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_explain)

    importance = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    print("\nTop 10 SHAP features:")
    print(importance.head(10).to_string())

    importance.head(15).sort_values().plot(
        kind="barh",
        figsize=(10, 7),
        title="CareFlow AI — Top SHAP Features",
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_feature_importance.png",
        dpi=150,
    )
    plt.close()

    print(
        "\nSaved:",
        output_dir / "shap_feature_importance.png",
    )


if __name__ == "__main__":
    main()
