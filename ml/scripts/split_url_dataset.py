import os
import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_FILE = "ml/data/processed/url_features_dataset.csv"

OUTPUT_DIR = "ml/data/processed"


TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_url.csv"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "val_url.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test_url.csv"
)


RANDOM_STATE = 42


def main():

    print("=" * 70)
    print("SPLITTING URL FEATURE DATASET")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Total samples: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print("\nLabel distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # First split:
    # 80% train
    # 20% temporary
    # ---------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # ---------------------------------------------------------
    # Second split:
    # Temporary -> 50% validation, 50% test
    #
    # Final:
    # 80% train
    # 10% validation
    # 10% test
    # ---------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    # Reset indexes
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(
        f"\nTraining samples:   {len(train_df):,}"
    )

    print(
        f"Validation samples: {len(val_df):,}"
    )

    print(
        f"Test samples:       {len(test_df):,}"
    )

    print("\nTraining labels:")

    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation labels:")

    print(
        val_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nTest labels:")

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Check for duplicate rows
    # ---------------------------------------------------------

    print("\nChecking duplicate rows...")

    print(
        f"Train duplicates: {train_df.duplicated().sum():,}"
    )

    print(
        f"Validation duplicates: {val_df.duplicated().sum():,}"
    )

    print(
        f"Test duplicates: {test_df.duplicated().sum():,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    val_df.to_csv(
        VAL_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print("\nFiles saved:")

    print(
        f"Train: {TRAIN_FILE}"
    )

    print(
        f"Validation: {VAL_FILE}"
    )

    print(
        f"Test: {TEST_FILE}"
    )

    print("\n" + "=" * 70)
    print("URL DATASET SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()