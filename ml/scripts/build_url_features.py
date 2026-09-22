import os
import pandas as pd
from tqdm import tqdm

from url_features import (
    extract_url_features,
    URL_FEATURES
)


INPUT_FILE = "ml/data/processed/url_dataset.csv"

OUTPUT_FILE = (
    "ml/data/processed/"
    "url_features_dataset.csv"
)


def main():

    print("=" * 70)
    print("BUILDING URL FEATURE DATASET")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input samples: {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns)}"
    )

    print("\nExtracting URL features...")

    feature_rows = []

    for url in tqdm(
        df["URL"].astype(str),
        total=len(df),
        desc="Processing URLs"
    ):

        features = extract_url_features(
            url
        )

        feature_rows.append(
            features
        )

    features_df = pd.DataFrame(
        feature_rows,
        columns=URL_FEATURES
    )

    result = features_df.copy()

    result["label"] = (
        df["label"]
        .astype(int)
        .values
    )

    print("\nFeature dataset created.")

    print(
        f"Shape: {result.shape}"
    )

    print("\nLabels:")

    print(
        result["label"]
        .value_counts()
        .sort_index()
    )

    print("\nChecking missing values...")

    missing = (
        result
        .isnull()
        .sum()
        .sum()
    )

    print(
        f"Missing values: {missing}"
    )

    print("\nSaving dataset...")

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nFirst 5 rows:")

    print(
        result.head().to_string()
    )

    print("\n" + "=" * 70)
    print("URL FEATURE DATASET COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()