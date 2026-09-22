import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


FEATURE_DIR = "ml/data/processed"

SVD_DIR = "ml/data/processed/url_svd"

MODEL_DIR = "ml/models/xgboost_url"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "url_xgboost_improved.json"
)

RESULTS_FILE = os.path.join(
    MODEL_DIR,
    "test_results_improved.json"
)


HANDCRAFTED_FEATURES = [
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
    "NoOfExternalRef",
]


def load_features():

    print("\nLoading handcrafted features...")

    train_df = pd.read_csv(
        os.path.join(
            FEATURE_DIR,
            "train_url_features.csv"
        )
    )

    val_df = pd.read_csv(
        os.path.join(
            FEATURE_DIR,
            "val_url_features.csv"
        )
    )

    test_df = pd.read_csv(
        os.path.join(
            FEATURE_DIR,
            "test_url_features.csv"
        )
    )

    X_train_hand = train_df[
        HANDCRAFTED_FEATURES
    ].values.astype(np.float32)

    X_val_hand = val_df[
        HANDCRAFTED_FEATURES
    ].values.astype(np.float32)

    X_test_hand = test_df[
        HANDCRAFTED_FEATURES
    ].values.astype(np.float32)

    y_train = train_df[
        "label"
    ].values.astype(np.int32)

    y_val = val_df[
        "label"
    ].values.astype(np.int32)

    y_test = test_df[
        "label"
    ].values.astype(np.int32)

    print(
        f"Train handcrafted: {X_train_hand.shape}"
    )

    print(
        f"Validation handcrafted: {X_val_hand.shape}"
    )

    print(
        f"Test handcrafted: {X_test_hand.shape}"
    )

    return (
        X_train_hand,
        X_val_hand,
        X_test_hand,
        y_train,
        y_val,
        y_test
    )


def load_svd():

    print("\nLoading SVD features...")

    X_train_svd = np.load(
        os.path.join(
            SVD_DIR,
            "train_svd.npy"
        )
    )

    X_val_svd = np.load(
        os.path.join(
            SVD_DIR,
            "val_svd.npy"
        )
    )

    X_test_svd = np.load(
        os.path.join(
            SVD_DIR,
            "test_svd.npy"
        )
    )

    print(
        f"Train SVD: {X_train_svd.shape}"
    )

    print(
        f"Validation SVD: {X_val_svd.shape}"
    )

    print(
        f"Test SVD: {X_test_svd.shape}"
    )

    return (
        X_train_svd,
        X_val_svd,
        X_test_svd
    )


def main():

    print("=" * 70)
    print("IMPROVED XGBOOST URL PHISHING DETECTOR")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load handcrafted features
    # ---------------------------------------------------------

    (
        X_train_hand,
        X_val_hand,
        X_test_hand,
        y_train,
        y_val,
        y_test
    ) = load_features()

    # ---------------------------------------------------------
    # Load SVD features
    # ---------------------------------------------------------

    (
        X_train_svd,
        X_val_svd,
        X_test_svd
    ) = load_svd()

    # ---------------------------------------------------------
    # Combine features
    # ---------------------------------------------------------

    print("\nCombining features...")

    X_train = np.hstack(
        [
            X_train_hand,
            X_train_svd
        ]
    )

    X_val = np.hstack(
        [
            X_val_hand,
            X_val_svd
        ]
    )

    X_test = np.hstack(
        [
            X_test_hand,
            X_test_svd
        ]
    )

    print(
        f"Final train shape: {X_train.shape}"
    )

    print(
        f"Final validation shape: {X_val.shape}"
    )

    print(
        f"Final test shape: {X_test.shape}"
    )

    print("\nTraining labels:")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    print("\nCreating XGBoost model...")

    model = xgb.XGBClassifier(

        n_estimators=500,

        max_depth=8,

        learning_rate=0.05,

        subsample=0.85,

        colsample_bytree=0.8,

        min_child_weight=2,

        gamma=0,

        reg_alpha=0.0,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        device="cuda",

        random_state=42,

        n_jobs=4
    )

    print("\nStarting GPU training...")

    print(
        "Device: CUDA"
    )

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_train,
                y_train
            ),
            (
                X_val,
                y_val
            )
        ],

        verbose=10
    )

    # ---------------------------------------------------------
    # Test prediction
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATING TEST SET")
    print("=" * 70)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\n" + "=" * 70)
    print("IMPROVED XGBOOST TEST RESULTS")
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
        f"Actual Legit     "
        f"{cm[0][0]:6d} "
        f"{cm[0][1]:9d}"
    )

    print(
        f"Actual Phishing  "
        f"{cm[1][0]:6d} "
        f"{cm[1][1]:9d}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "LEGITIMATE",
                "PHISHING"
            ],
            digits=4
        )
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    print("\nSaving model...")

    model.save_model(
        MODEL_FILE
    )

    print(
        f"Model saved to:"
    )

    print(
        MODEL_FILE
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {

        "model": "XGBoost",

        "representation": (
            "25 handcrafted URL features "
            "+ 300 SVD character features"
        ),

        "train_samples": int(
            len(y_train)
        ),

        "validation_samples": int(
            len(y_val)
        ),

        "test_samples": int(
            len(y_test)
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "confusion_matrix": (
            cm.tolist()
        )
    }

    with open(
        RESULTS_FILE,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print("\n" + "=" * 70)
    print("IMPROVED URL MODEL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()