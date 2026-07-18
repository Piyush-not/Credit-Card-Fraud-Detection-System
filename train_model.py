"""
Credit Card Fraud Detection - Training & Evaluation
----------------------------------------------------
Trains two classifiers (Logistic Regression, Random Forest) on the
Kaggle credit card fraud dataset (creditcard.csv: Time, V1-V28, Amount, Class),
handles the severe class imbalance via class-weighting, evaluates with
imbalance-appropriate metrics (ROC-AUC, PR-AUC, precision/recall/F1),
and saves the trained model + plots + metrics to disk.

Usage:
    python train_model.py --data data/creditcard.csv
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(int)
    return df


def train_and_evaluate(df: pd.DataFrame, outputs_dir: str, models_dir: str):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Only Time/Amount need scaling; V1-V28 are already PCA components.
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test_s[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}
    probs_by_model = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        probs_by_model[name] = probs

        roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, digits=4)

        results[name] = {
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist(),
            "precision_fraud": report["1"]["precision"],
            "recall_fraud": report["1"]["recall"],
            "f1_fraud": report["1"]["f1-score"],
        }

        print(f"=== {name} ===")
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
        print(classification_report(y_test, preds, digits=4))

        joblib.dump(model, os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.joblib"))

    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    with open(os.path.join(outputs_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance (Random Forest)
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )
    importances.to_csv(os.path.join(outputs_dir, "feature_importance.csv"))
    print("\nTop 10 features (Random Forest importance):")
    print(importances.head(10))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, probs in probs_by_model.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        axes[0].plot(rec, prec, label=name)
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[1].plot(fpr, tpr, label=name)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "curves.png"), dpi=130)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    importances.head(12).sort_values().plot(kind="barh", ax=ax2, color="#4C72B0")
    ax2.set_title("Top 12 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "feature_importance.png"), dpi=130)
    plt.close(fig2)

    print(f"\nSaved models to '{models_dir}/' and plots/metrics to '{outputs_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit card fraud detection models.")
    parser.add_argument(
        "--data", default="data/creditcard.csv", help="Path to creditcard.csv"
    )
    parser.add_argument("--outputs", default="outputs", help="Directory for plots/metrics")
    parser.add_argument("--models", default="models", help="Directory for saved models")
    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    df = load_data(args.data)
    print(f"Loaded {len(df):,} transactions | {df['Class'].sum()} fraudulent "
          f"({100 * df['Class'].mean():.3f}%)")

    train_and_evaluate(df, args.outputs, args.models)
