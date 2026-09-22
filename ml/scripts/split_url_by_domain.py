import os
import pandas as pd
import tldextract

from sklearn.model_selection import StratifiedGroupKFold


INPUT_FILE = "ml/data/processed/url_dataset.csv"

OUTPUT_DIR = "ml/data/processed"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_url_domain.csv"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "val_url_domain.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test_url_domain.csv"
)


RANDOM_STATE = 42


def get_domain(url):

    try:

        extracted = tldextract.extract(
            str(url)
        )

        if extracted.domain and extracted.suffix:

            return (
                extracted.domain
                + "."
                + extracted.suffix
            )

        return extracted.domain

    except Exception:

        return ""


def main():

    print("=" * 70)
    print("DOMAIN-AWARE URL DATASET SPLIT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Total URLs: {len(df):,}"
    )

    print("\nExtracting registered domains...")

    df["domain_group"] = (
        df["URL"]
        .astype(str)
        .apply(get_domain)
    )

    empty_domains = (
        df["domain_group"]
        .eq("")
        .sum()
    )

    print(
        f"Empty domains: {empty_domains:,}"
    )

    print(
        f"Unique domains: "
        f"{df['domain_group'].nunique():,}"
    )

    # ---------------------------------------------------------
    # First split:
    # 80% train
    # 20% temporary
    #
    # StratifiedGroupKFold ensures:
    # - same domain stays together
    # - labels remain approximately balanced
    # ---------------------------------------------------------

    print("\nCreating domain-aware folds...")

    splitter_1 = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            df,
            y=df["label"],
            groups=df["domain_group"]
        )
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    temp_df = df.iloc[
        temp_idx
    ].copy()

    # ---------------------------------------------------------
    # Split temporary 50/50:
    #
    # 10% validation
    # 10% test
    # ---------------------------------------------------------

    splitter_2 = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    val_idx, test_idx = next(
        splitter_2.split(
            temp_df,
            y=temp_df["label"],
            groups=temp_df["domain_group"]
        )
    )

    val_df = temp_df.iloc[
        val_idx
    ].copy()

    test_df = temp_df.iloc[
        test_idx
    ].copy()

    # ---------------------------------------------------------
    # Remove helper column
    # ---------------------------------------------------------

    train_df = train_df.drop(
        columns=["domain_group"]
    )

    val_df = val_df.drop(
        columns=["domain_group"]
    )

    test_df = test_df.drop(
        columns=["domain_group"]
    )

    # ---------------------------------------------------------
    # Reset indexes
    # ---------------------------------------------------------

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(
        f"\nTraining URLs:   {len(train_df):,}"
    )

    print(
        f"Validation URLs: {len(val_df):,}"
    )

    print(
        f"Test URLs:       {len(test_df):,}"
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
    # Verify domain separation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFYING DOMAIN SEPARATION")
    print("=" * 70)

    train_domains = set(
        train_df["URL"]
        .astype(str)
        .apply(get_domain)
    )

    val_domains = set(
        val_df["URL"]
        .astype(str)
        .apply(get_domain)
    )

    test_domains = set(
        test_df["URL"]
        .astype(str)
        .apply(get_domain)
    )

    train_val_overlap = (
        train_domains & val_domains
    )

    train_test_overlap = (
        train_domains & test_domains
    )

    val_test_overlap = (
        val_domains & test_domains
    )

    print(
        f"Train ∩ Validation domains: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test domains: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test domains: "
        f"{len(val_test_overlap)}"
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
    print("DOMAIN-AWARE SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()