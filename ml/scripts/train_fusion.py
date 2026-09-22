import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


INPUT_FILE = "ml/data/processed/fusion_predictions.csv"

MODEL_DIR = "ml/models/fusion"

MODEL_FILE = f"{MODEL_DIR}/fusion_model.joblib"

RESULTS_FILE = f"{MODEL_DIR}/fusion_results.json"


FEATURES = [
    "deberta_probability",
    "url_probability",
    "url_count"
]


def evaluate(model, X, y, name):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 70)
    print(f"{name} RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix")

    print("                 Predicted")
    print("                 Legit  Phishing")

    print(
        f"Actual Legit     "
        f"{cm[0][0]:6d}  "
        f"{cm[0][1]:8d}"
    )

    print(
        f"Actual Phishing  "
        f"{cm[1][0]:6d}  "
        f"{cm[1][1]:8d}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "LEGITIMATE",
                "PHISHING"
            ],
            digits=4
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }


def main():

    print("=" * 70)
    print("DEBERTA + XGBOOST FUSION MODEL")
    print("=" * 70)

    print("\nLoading fusion predictions...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Total samples: {len(df):,}"
    )

    print("\nFeatures:")

    print(
        FEATURES
    )

    X = df[FEATURES].copy()

    y = df["label"].astype(int)

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=42
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=42
        )
    )

    print("\nDataset split:")

    print(
        f"Training:   {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_val):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    print("\nTraining labels:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    print("\nValidation labels:")

    print(
        y_val.value_counts()
        .sort_index()
    )

    print("\nTest labels:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    print("\nCreating fusion model...")

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    print("\nTraining fusion model...")

    model.fit(
        X_train,
        y_train
    )

    print("Fusion model trained.")

    validation_results = evaluate(
        model,
        X_val,
        y_val,
        "FUSION VALIDATION"
    )

    test_results = evaluate(
        model,
        X_test,
        y_test,
        "FUSION TEST"
    )

    print("\n" + "=" * 70)
    print("FUSION MODEL COEFFICIENTS")
    print("=" * 70)

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    for feature, coefficient in zip(
        FEATURES,
        classifier.coef_[0]
    ):

        print(
            f"{feature:25s}: "
            f"{coefficient:.6f}"
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    import joblib

    joblib.dump(
        model,
        MODEL_FILE
    )

    results = {
        "model": "Logistic Regression Late Fusion",
        "features": FEATURES,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "validation": validation_results,
        "test": test_results,
        "coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(
                FEATURES,
                classifier.coef_[0]
            )
        }
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nModel saved to:")
    print(MODEL_FILE)

    print("\nResults saved to:")
    print(RESULTS_FILE)

    print("\n" + "=" * 70)
    print("FUSION TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()