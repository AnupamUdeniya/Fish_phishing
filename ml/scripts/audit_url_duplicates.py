import pandas as pd


ORIGINAL_FILE = "ml/data/processed/url_dataset.csv"
FEATURE_FILE = "ml/data/processed/url_features_dataset.csv"


def main():

    print("=" * 70)
    print("URL DUPLICATE / FEATURE COLLISION AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    print("\nLoading original URL dataset...")

    original = pd.read_csv(
        ORIGINAL_FILE
    )

    print(
        f"Original rows: {len(original):,}"
    )

    print("\nLoading feature dataset...")

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    # ---------------------------------------------------------
    # 1. Exact URL duplicates
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. EXACT URL DUPLICATES")
    print("=" * 70)

    unique_urls = (
        original["URL"]
        .astype(str)
        .nunique()
    )

    duplicate_urls = (
        len(original) - unique_urls
    )

    print(
        f"Unique URLs:       {unique_urls:,}"
    )

    print(
        f"Duplicate URL rows: {duplicate_urls:,}"
    )

    # ---------------------------------------------------------
    # 2. Conflicting URL labels
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. DUPLICATE URL LABEL CONSISTENCY")
    print("=" * 70)

    url_label_counts = (
        original
        .groupby("URL")["label"]
        .nunique()
    )

    conflicting_urls = (
        (url_label_counts > 1)
        .sum()
    )

    print(
        f"URLs with multiple labels: "
        f"{conflicting_urls:,}"
    )

    # ---------------------------------------------------------
    # 3. Feature vector duplicates
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. FEATURE VECTOR DUPLICATES")
    print("=" * 70)

    feature_columns = [
        column
        for column in features.columns
        if column != "label"
    ]

    unique_feature_vectors = (
        features[feature_columns]
        .drop_duplicates()
        .shape[0]
    )

    duplicate_feature_rows = (
        len(features)
        - unique_feature_vectors
    )

    print(
        f"Unique feature vectors: "
        f"{unique_feature_vectors:,}"
    )

    print(
        f"Duplicate feature rows: "
        f"{duplicate_feature_rows:,}"
    )

    # ---------------------------------------------------------
    # 4. Feature collisions with different labels
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. FEATURE VECTOR LABEL COLLISIONS")
    print("=" * 70)

    feature_label_counts = (
        features
        .groupby(feature_columns)["label"]
        .nunique()
    )

    conflicting_feature_vectors = (
        (feature_label_counts > 1)
        .sum()
    )

    print(
        f"Feature vectors with conflicting labels: "
        f"{conflicting_feature_vectors:,}"
    )

    # ---------------------------------------------------------
    # 5. Label distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. LABEL DISTRIBUTION")
    print("=" * 70)

    print(
        features["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 6. Feature uniqueness ratio
    # ---------------------------------------------------------

    uniqueness_ratio = (
        unique_feature_vectors /
        len(features)
    )

    print("\n" + "=" * 70)
    print("6. FEATURE UNIQUENESS")
    print("=" * 70)

    print(
        f"Unique feature ratio: "
        f"{uniqueness_ratio:.4f}"
    )

    print(
        f"Unique feature percentage: "
        f"{uniqueness_ratio * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()