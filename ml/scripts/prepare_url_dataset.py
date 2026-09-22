from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "phiusiil"
PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_FILE = PROCESSED / "url_dataset.csv"

PROCESSED.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 80)
    print("PHIUSIIL URL DATASET PREPARATION")
    print("=" * 80)

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    required_columns = [
        "URL",
        "Domain",
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
        "DegitRatioInURL",
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
        "label"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing columns:")
        for col in missing_columns:
            print(f"  - {col}")
        return

    df = df[required_columns].copy()

    df["URL"] = df["URL"].fillna("").astype(str)
    df["Domain"] = df["Domain"].fillna("").astype(str)

    df = df.drop_duplicates(
        subset=["URL", "label"]
    )

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = df["label"].astype(int)

    print("\nOriginal PhiUSIIL labels:")
    print(df["label"].value_counts().sort_index())

    df["label"] = df["label"].map({
        1: 0,
        0: 1
    })

    print("\nOur project labels:")
    print("0 = LEGITIMATE")
    print("1 = PHISHING")
    print()
    print(df["label"].value_counts().sort_index())

    numeric_columns = [
        col for col in required_columns
        if col not in ["URL", "Domain", "label"]
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[numeric_columns] = df[numeric_columns].fillna(0)

    df = df.reset_index(drop=True)

    df.insert(
        0,
        "id",
        range(1, len(df) + 1)
    )

    df.insert(
        1,
        "source",
        "PhiUSIIL"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFinal dataset:")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("URL DATASET PREPARATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()