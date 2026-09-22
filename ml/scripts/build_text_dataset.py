from pathlib import Path
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

EMAIL_FILE = PROCESSED / "email_dataset.csv"
SPAPHISH_FILE = PROCESSED / "spaphish_dataset.csv"
SHORT_TEXT_FILE = PROCESSED / "short_text_dataset.csv"

FINAL_FILE = PROCESSED / "text_dataset.csv"

TRAIN_FILE = PROCESSED / "train_text.csv"
VAL_FILE = PROCESSED / "val_text.csv"
TEST_FILE = PROCESSED / "test_text.csv"


RANDOM_STATE = 42


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_for_duplicate(text):
    text = clean_text(text).lower()

    text = " ".join(text.split())

    return text


def create_hash(text):
    normalized = normalize_for_duplicate(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_email():

    print("\nLoading main email dataset...")

    df = pd.read_csv(
        EMAIL_FILE,
        low_memory=False
    )

    result = pd.DataFrame()

    result["source"] = df["source"].astype(str)
    result["text"] = df["text"].apply(clean_text)
    result["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    return result


def load_spaphish():

    print("Loading SpaPhish dataset...")

    df = pd.read_csv(
        SPAPHISH_FILE,
        low_memory=False
    )

    result = pd.DataFrame()

    result["source"] = ["SpaPhish"] * len(df)
    result["text"] = df["text"].apply(clean_text)
    result["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    return result


def load_short_text():

    print("Loading short-text dataset...")

    df = pd.read_csv(
        SHORT_TEXT_FILE,
        low_memory=False
    )

    result = pd.DataFrame()

    result["source"] = ["ShortText"] * len(df)
    result["text"] = df["text"].apply(clean_text)
    result["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    return result


def print_dataset_stats(df, name):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Rows: {len(df):,}")

    print("\nLabels:")

    labels = df["label"].value_counts().sort_index()

    for label, count in labels.items():

        if label == 0:
            name_label = "LEGITIMATE"
        elif label == 1:
            name_label = "PHISHING"
        else:
            name_label = "UNKNOWN"

        print(
            f"{int(label)} = {name_label}: {count:,}"
        )

    print("\nSources:")

    print(
        df["source"]
        .value_counts()
        .to_string()
    )


def main():

    print("=" * 80)
    print("FINAL TEXT DATASET BUILDER")
    print("=" * 80)

    email_df = load_email()
    spaphish_df = load_spaphish()
    short_text_df = load_short_text()

    print("\nCombining datasets...")

    df = pd.concat(
        [
            email_df,
            spaphish_df,
            short_text_df
        ],
        ignore_index=True
    )

    print(f"Combined rows: {len(df):,}")

    print("\nRemoving invalid labels...")

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df["label"] = df["label"].astype(int)

    print(f"Rows after label filtering: {len(df):,}")

    print("\nRemoving empty messages...")

    df = df[
        df["text"].str.strip().str.len() > 0
    ].copy()

    print(f"Rows after empty-text removal: {len(df):,}")

    print("\nDetecting duplicate messages...")

    df["text_hash"] = df["text"].apply(
        create_hash
    )

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["text_hash"],
        keep="first"
    ).copy()

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Duplicate messages removed: "
        f"{duplicates_removed:,}"
    )

    df = df.drop(
        columns=["text_hash"]
    )

    df = df.reset_index(drop=True)

    df.insert(
        0,
        "id",
        range(1, len(df) + 1)
    )

    print_dataset_stats(
        df,
        "FINAL COMBINED TEXT DATASET"
    )

    print("\nSaving final text dataset...")

    df.to_csv(
        FINAL_FILE,
        index=False
    )

    print(f"Saved: {FINAL_FILE}")

    print("\nCreating train/validation/test split...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

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

    print("\n" + "=" * 80)
    print("DATA SPLIT RESULTS")
    print("=" * 80)

    print(f"\nTrain:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    print("\nTRAIN labels:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nVALIDATION labels:")
    print(
        val_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nTEST labels:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nFiles created:")

    print(f"  {FINAL_FILE.name}")
    print(f"  {TRAIN_FILE.name}")
    print(f"  {VAL_FILE.name}")
    print(f"  {TEST_FILE.name}")

    print("\n" + "=" * 80)
    print("TEXT DATASET BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()