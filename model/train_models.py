import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET_COL = "target"
HERE = Path(__file__).resolve().parent          # .../model
REPO_ROOT = HERE.parent                          # repo root


def load_dataset():
    """Return (dataframe, target_column_name). Dataset-specific — see header."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns={"target": TARGET_COL})
    return df, TARGET_COL


def build_models():
    """The five classifiers required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }


def score_model(model, X, y_true):
    """All six evaluation metrics for a fitted binary classifier."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

FILE_STEMS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}


def main():
    df, target_col = load_dataset()
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"Dataset: {len(df)} instances, {len(feature_cols)} features "
          f"(binary target '{target_col}')")

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train_s, y_train)
        metrics = score_model(model, X_test_s, y_test)
        rows.append({"Model": name, **metrics})
        joblib.dump(model, HERE / f"{FILE_STEMS[name]}.pkl")
        print(f"  trained {name:<20} acc={metrics['Accuracy']:.4f} "
              f"auc={metrics['AUC']:.4f} mcc={metrics['MCC']:.4f}")

    joblib.dump(scaler, HERE / "scaler.pkl")
    (HERE / "feature_names.json").write_text(
        json.dumps({"target": target_col, "features": feature_cols}, indent=2)
    )

    # Comparison table for the README
    metrics_df = pd.DataFrame(rows).round(4)
    metrics_df.to_csv(HERE / "metrics.csv", index=False)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[target_col] = y_test
    test_df.to_csv(REPO_ROOT / "test_data.csv", index=False)

    print("\nComparison table:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved test_data.csv with {len(test_df)} rows to repo root.")


if __name__ == "__main__":
    main()
