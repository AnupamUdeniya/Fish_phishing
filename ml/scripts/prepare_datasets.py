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

    phishing = {
        "1",
        "phishing",
        "phish",
        "malicious",
        "malware",
        "true",
        "yes"
    }

    legitimate = {
        "0",
        "legitimate",
        "legit",
        "ham",
        "benign",
        "safe",
        "false",
        "no"
    }

    if value in phishing:
        return 1

    if value in legitimate:
        return 0

    return None


def combine_text(row):
    subject = clean_text(row.get("subject", ""))
    body = clean_text(row.get("body", ""))
    sender = clean_text(row.get("sender", ""))

    parts = []

    if sender:
        parts.append("Sender: " + sender)

    if subject:
        parts.append("Subject: " + subject)

    if body:
        parts.append("Body: " + body)

    return "\n".join(parts)


def process_email_file(path, source):
    print(f"Processing: {path.name}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "label" not in df.columns:
        print("  Skipping: no label column")
        return pd.DataFrame()

    result = pd.DataFrame()

    result["source"] = [source] * len(df)
    result["subject"] = (
        df["subject"].apply(clean_text)
        if "subject" in df.columns
        else ""
    )
    result["sender"] = (
        df["sender"].apply(clean_text)
        if "sender" in df.columns
        else ""
    )
    result["body"] = (
        df["body"].apply(clean_text)
        if "body" in df.columns
        else ""
    )
    result["url"] = (
        df["urls"].apply(clean_text)
        if "urls" in df.columns
        else ""
    )

    result["text"] = df.apply(combine_text, axis=1)

    result["label"] = df["label"].apply(normalize_label)

    result = result.dropna(subset=["label"])
    result["label"] = result["label"].astype(int)

    return result


def process_spaphish(path):
    print(f"Processing SpaPhish: {path.name}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    print("  Columns found:")
    print("  ", list(df.columns))

    return pd.DataFrame()


def process_short_text(path):
    print(f"Processing short text: {path.name}")

    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    print("  Columns found:")
    print("  ", list(df.columns))

    return pd.DataFrame()


def main():

    print("=" * 80)
    print("PHISHING DATASET PREPARATION")
    print("=" * 80)

    email_frames = []

    curated = RAW / "curated_email_11"

    for path in sorted(curated.glob("*.csv")):

        if "_vectorized" in path.name.lower():
            continue

        frame = process_email_file(
            path,
            f"curated_{path.stem}"
        )

        if not frame.empty:
            email_frames.append(frame)

    if email_frames:

        email_df = pd.concat(
            email_frames,
            ignore_index=True
        )

        email_df = email_df.drop_duplicates(
            subset=["text", "label"]
        )

        email_output = PROCESSED / "email_dataset.csv"

        email_df.to_csv(
            email_output,
            index=False
        )

        print("\nEMAIL DATASET")
        print("-" * 40)
        print(f"Rows: {len(email_df):,}")
        print("\nLabels:")
        print(email_df["label"].value_counts().sort_index())
        print(f"\nSaved to:")
        print(email_output)

    spaphish = RAW / "spaphish"

    spaphish_files = list(
        spaphish.glob("*.csv")
    )
    for path in spaphish_files:
        process_spaphish(path)

    short_text = RAW / "short_text"

    for path in short_text.glob("*"):

        if path.suffix.lower() in [".xlsx", ".xls"]:
            process_short_text(path)

    print("\n" + "=" * 80)
    print("PREPARATION STAGE 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()