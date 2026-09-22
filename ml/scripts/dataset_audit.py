from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def inspect_file(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            return

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing):
            print(missing.to_string())
        else:
            print("  None")

        print("\nPossible label columns:")

        keywords = [
            "label",
            "class",
            "target",
            "type",
            "category",
            "spam",
            "phish",
            "legitimate"
        ]

        found = False

        for col in df.columns:
            name = str(col).lower()

            if any(keyword in name for keyword in keywords):
                found = True
                print(f"\n  {col}")

                values = df[col].value_counts(dropna=False).head(20)
                print(values.to_string())

        if not found:
            print("  No obvious label column found.")

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(max_cols=8))

    except Exception as e:
        print(f"ERROR: {e}")


def main():
    print("=" * 80)
    print("PHISHING DATASET AUDIT")
    print("=" * 80)

    print(f"\nRaw data directory:")
    print(RAW)

    if not RAW.exists():
        print("\nERROR: Raw data directory does not exist.")
        return

    files = []

    for folder in RAW.iterdir():
        if folder.is_dir():
            for file in folder.rglob("*"):
                if file.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                    files.append(file)

    if not files:
        print("\nNo CSV/XLSX datasets found.")
        return

    print(f"\nFound {len(files)} dataset files.")

    for file in sorted(files):
        inspect_file(file)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()