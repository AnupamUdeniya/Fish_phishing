from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    phishing_labels = {
        "1",
        "phishing",
        "phish",
        "malicious",
        "malware",
        "true",
        "yes"
    }

    legitimate_labels = {
        "0",
        "legitimate",
        "legit",
        "ham",
        "benign",
        "safe",
        "false",
        "no"
    }

    if value in phishing_labels:
        return 1

    if value in legitimate_labels:
        return 0

    return None


def prepare_spaphish():

    print("\n" + "=" * 80)
    print("PROCESSING SPAPHISH")
    print("=" * 80)

    folder = RAW / "spaphish"

    csv_files = list(folder.rglob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV file found.")
        return

    input_file = None

    for file in csv_files:
        if "spaphish dataset" in file.name.lower():
            input_file = file
            break

    if input_file is None:
        input_file = csv_files[0]

    print(f"Loading: {input_file.name}")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    columns_lower = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    subject_col = columns_lower.get("subject")
    body_col = columns_lower.get("body")
    url_col = columns_lower.get("urls")

    label_col = None

    for key, original in columns_lower.items():
        if key in {
            "label",
            "labels",
            "class",
            "target"
        }:
            label_col = original
            break

    if label_col is None:
        print("\nERROR: Could not find label column.")
        return

    result = pd.DataFrame()

    result["source"] = ["SpaPhish"] * len(df)

    if subject_col:
        result["subject"] = df[subject_col].apply(clean_text)
    else:
        result["subject"] = ""

    if body_col:
        result["body"] = df[body_col].apply(clean_text)
    else:
        result["body"] = ""

    if url_col:
        result["url"] = df[url_col].apply(clean_text)
    else:
        result["url"] = ""

    result["text"] = (
        "Subject: "
        + result["subject"]
        + "\nBody: "
        + result["body"]
    )

    result["label"] = df[label_col].apply(
        normalize_label
    )

    result = result.dropna(
        subset=["label"]
    )

    result["label"] = result["label"].astype(int)

    result = result[
        result["text"].str.strip().str.len() > 0
    ]

    result = result.drop_duplicates(
        subset=["text", "label"]
    )

    result = result.reset_index(drop=True)

    result.insert(
        0,
        "id",
        range(1, len(result) + 1)
    )

    output = PROCESSED / "spaphish_dataset.csv"

    result.to_csv(
        output,
        index=False
    )

    print("\nFinal SpaPhish dataset:")
    print(f"Rows: {len(result):,}")

    print("\nLabels:")
    print(
        result["label"]
        .value_counts()
        .sort_index()
    )

    print("\n0 = LEGITIMATE")
    print("1 = PHISHING")

    print(f"\nSaved to:")
    print(output)


def prepare_short_text():

    print("\n" + "=" * 80)
    print("PROCESSING SHORT TEXT DATASET")
    print("=" * 80)

    folder = RAW / "short_text"

    excel_files = list(folder.glob("*.xlsx"))

    if not excel_files:
        print("ERROR: No XLSX file found.")
        return

    input_file = excel_files[0]

    print(f"Loading: {input_file.name}")

    df = pd.read_excel(input_file)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    columns_lower = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    text_col = None

    for name in [
        "normalized_text",
        "raw_text",
        "text",
        "message"
    ]:
        if name in columns_lower:
            text_col = columns_lower[name]
            break

    label_col = None

    for name in [
        "labels",
        "label",
        "class",
        "target"
    ]:
        if name in columns_lower:
            label_col = columns_lower[name]
            break

    source_col = None

    for name in [
        "source_type",
        "source",
        "type"
    ]:
        if name in columns_lower:
            source_col = columns_lower[name]
            break

    if text_col is None:
        print("\nERROR: Could not find text column.")
        return

    if label_col is None:
        print("\nERROR: Could not find label column.")
        return

    result = pd.DataFrame()

    result["source"] = ["ShortText"] * len(df)

    result["text"] = df[text_col].apply(
        clean_text
    )

    result["raw_text"] = (
        df[columns_lower["raw_text"]].apply(clean_text)
        if "raw_text" in columns_lower
        else result["text"]
    )

    if source_col:
        result["source_type"] = (
            df[source_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        result["source_type"] = ""

    result["label"] = df[label_col].apply(
        normalize_label
    )

    result = result.dropna(
        subset=["label"]
    )

    result["label"] = result["label"].astype(int)

    result = result[
        result["text"].str.strip().str.len() > 0
    ]

    result = result.drop_duplicates(
        subset=["text", "label"]
    )

    result = result.reset_index(drop=True)

    result.insert(
        0,
        "id",
        range(1, len(result) + 1)
    )

    output = PROCESSED / "short_text_dataset.csv"

    result.to_csv(
        output,
        index=False
    )

    print("\nFinal short-text dataset:")
    print(f"Rows: {len(result):,}")

    print("\nLabels:")
    print(
        result["label"]
        .value_counts()
        .sort_index()
    )

    print("\n0 = LEGITIMATE")
    print("1 = PHISHING")

    print(f"\nSaved to:")
    print(output)


def main():

    print("=" * 80)
    print("OTHER DATASET PREPARATION")
    print("=" * 80)

    prepare_spaphish()
    prepare_short_text()

    print("\n" + "=" * 80)
    print("OTHER DATASETS PREPARATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()