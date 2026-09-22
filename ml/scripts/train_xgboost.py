import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


DATA_FILE = "ml/data/processed/url_dataset.csv"
OUTPUT_DIR = "ml/models/xgboost_url"

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

N_ESTIMATORS = 500
MAX_DEPTH = 8
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("XGBOOST URL PHISHING DETECTOR")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print("\nLabels:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    feature_columns = [
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        "NoOfSubDomain",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "NoOfDegitsInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL",
        "IsHTTPS",
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasPasswordField",
        "Bank",
        "Pay",
        "Crypto",
        "NoOfExternalRef"
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns: "
            + str(missing_features)
        )

    X = df[
        feature_columns
    ].copy()

    y = df["label"].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    print("\nFeatures used:")

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    print("\nCreating train/validation/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    validation_ratio = (
        VALIDATION_SIZE /
        (1.0 - TEST_SIZE)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print(
        f"Training samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_val):,}"
    )

    print(
        f"Test samples:       {len(X_test):,}"
    )

    print("\nLabel distribution:")

    print(
        f"Train:\n{y_train.value_counts().sort_index()}"
    )

    print(
        f"\nValidation:\n{y_val.value_counts().sort_index()}"
    )

    print(
        f"\nTest:\n{y_test.value_counts().sort_index()}"
    )

    print("\nCreating XGBoost model...")

    model = xgb.XGBClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        device="cuda",

        random_state=RANDOM_STATE,

        n_jobs=1
    )

    print("\nStarting GPU training...")

    print(
        "Device: CUDA"
    )

    print(
        "Tree method: hist"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=True
    )

    print("\nTraining complete.")

    print("\nEvaluating test set...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("XGBOOST TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        "                 Predicted"
    )

    print(
        "                 Legit  Phishing"
    )

    print(
        f"Actual Legit     {cm[0][0]:6d}  {cm[0][1]:8d}"
    )

    print(
        f"Actual Phishing  {cm[1][0]:6d}  {cm[1][1]:8d}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "LEGITIMATE",
                "PHISHING"
            ],
            digits=4
        )
    )

    print("\nTop URL Features")

    importances = pd.Series(
        model.feature_importances_,
        index=feature_columns
    )

    importances = (
        importances
        .sort_values(
            ascending=False
        )
    )

    print(
        importances.to_string()
    )

    model_file = os.path.join(
        OUTPUT_DIR,
        "url_xgboost.json"
    )

    model.save_model(
        model_file
    )

    print(
        f"\nModel saved to:"
    )

    print(
        model_file
    )

    feature_file = os.path.join(
        OUTPUT_DIR,
        "features.json"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=4
        )

    results = {
        "model": "XGBoost",
        "dataset": DATA_FILE,
        "features": feature_columns,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {
            key: float(value)
            for key, value
            in importances.items()
        }
    }

    results_file = os.path.join(
        OUTPUT_DIR,
        "test_results.json"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"Results saved to:"
    )

    print(
        results_file
    )

    print("\n" + "=" * 70)
    print("URL MODEL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()