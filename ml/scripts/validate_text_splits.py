from pathlib import Path
import pandas as pd
import hashlib
import re


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED / "train_text.csv"
VAL_FILE = PROCESSED / "val_text.csv"
TEST_FILE = PROCESSED / "test_text.csv"


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text):
    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()


def load_file(path):
    df = pd.read_csv(
        path,
        low_memory=False
    )

    required = [
        "id",
        "source",
        "text",
        "label"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    return df


def check_basic_quality(df, name):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")

    missing = df[
        ["id", "source", "text", "label"]
    ].isna().sum()

    print(missing.to_string())

    print("\nLabels:")

    labels = df["label"].value_counts().sort_index()

    for label, count in labels.items():

        if label == 0:
            label_name = "LEGITIMATE"
        elif label == 1:
            label_name = "PHISHING"
        else:
            label_name = "UNKNOWN"

        print(
            f"{int(label)} = {label_name}: {count:,}"
        )

    print("\nSources:")

    print(
        df["source"]
        .value_counts(dropna=False)
        .to_string()
    )

    text_lengths = df["text"].fillna("").astype(str).str.len()

    print("\nText length:")

    print(
        f"Minimum: {text_lengths.min():,}"
    )

    print(
        f"Maximum: {text_lengths.max():,}"
    )

    print(
        f"Mean:    {text_lengths.mean():,.1f}"
    )

    print(
        f"Median:  {text_lengths.median():,.1f}"
    )

    print(
        f"Empty:   {(text_lengths == 0).sum():,}"
    )

    print(
        f"< 20 characters: {(text_lengths < 20).sum():,}"
    )


def check_internal_duplicates(df, name):

    hashes = df["text"].apply(text_hash)

    duplicates = hashes.duplicated().sum()

    print(
        f"\n{name} duplicate messages: "
        f"{duplicates:,}"
    )

    return hashes


def check_cross_split_overlap(
    train_hashes,
    val_hashes,
    test_hashes
):

    train_set = set(train_hashes)
    val_set = set(val_hashes)
    test_set = set(test_hashes)

    train_val = len(
        train_set.intersection(val_set)
    )

    train_test = len(
        train_set.intersection(test_set)
    )

    val_test = len(
        val_set.intersection(test_set)
    )

    print("\n" + "=" * 70)
    print("CROSS-SPLIT EXACT OVERLAP")
    print("=" * 70)

    print(
        f"Train ∩ Validation: {train_val:,}"
    )

    print(
        f"Train ∩ Test:       {train_test:,}"
    )

    print(
        f"Validation ∩ Test:  {val_test:,}"
    )

    if (
        train_val == 0
        and train_test == 0
        and val_test == 0
    ):
        print("\nPASS: No exact text overlap between splits.")
    else:
        print(
            "\nWARNING: Exact text overlap detected."
        )


def check_id_overlap(train, val, test):

    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])

    print("\n" + "=" * 70)
    print("ID OVERLAP")
    print("=" * 70)

    print(
        f"Train ∩ Validation: "
        f"{len(train_ids.intersection(val_ids)):,}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_ids.intersection(test_ids)):,}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(val_ids.intersection(test_ids)):,}"
    )


def check_class_balance(train, val, test):

    print("\n" + "=" * 70)
    print("CLASS BALANCE")
    print("=" * 70)

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test)
    ]:

        counts = df["label"].value_counts(
            normalize=True
        ).sort_index()

        print(f"\n{name}")

        for label, percentage in counts.items():

            label_name = (
                "LEGITIMATE"
                if label == 0
                else "PHISHING"
            )

            print(
                f"{label_name}: "
                f"{percentage * 100:.2f}%"
            )


def check_source_balance(train, val, test):

    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION")
    print("=" * 70)

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test)
    ]:

        print(f"\n{name}:")

        print(
            df["source"]
            .value_counts()
            .to_string()
        )


def main():

    print("=" * 80)
    print("TEXT DATASET VALIDATION")
    print("=" * 80)

    print("\nLoading datasets...")

    train = load_file(TRAIN_FILE)
    val = load_file(VAL_FILE)
    test = load_file(TEST_FILE)

    print("\nAll files loaded successfully.")

    check_basic_quality(
        train,
        "TRAIN DATASET"
    )

    check_basic_quality(
        val,
        "VALIDATION DATASET"
    )

    check_basic_quality(
        test,
        "TEST DATASET"
    )

    train_hashes = check_internal_duplicates(
        train,
        "TRAIN"
    )

    val_hashes = check_internal_duplicates(
        val,
        "VALIDATION"
    )

    test_hashes = check_internal_duplicates(
        test,
        "TEST"
    )

    check_cross_split_overlap(
        train_hashes,
        val_hashes,
        test_hashes
    )

    check_id_overlap(
        train,
        val,
        test
    )

    check_class_balance(
        train,
        val,
        test
    )

    check_source_balance(
        train,
        val,
        test
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    total_rows = (
        len(train)
        + len(val)
        + len(test)
    )

    print(
        f"\nTotal rows across splits: "
        f"{total_rows:,}"
    )

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(val):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print("\nValidation finished.")

    print("=" * 80)


if __name__ == "__main__":
    main()