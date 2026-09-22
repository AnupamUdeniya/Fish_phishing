from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "TRAIN": PROCESSED / "train_text.csv",
    "VALIDATION": PROCESSED / "val_text.csv",
    "TEST": PROCESSED / "test_text.csv"
}


def analyze(name, path):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    df = pd.read_csv(
        path,
        low_memory=False
    )

    text = df["text"].fillna("").astype(str)

    lengths = text.str.len()

    print(f"\nRows: {len(df):,}")

    print("\nCharacter length statistics:")
    print(f"Minimum: {lengths.min():,}")
    print(f"25%:     {lengths.quantile(0.25):,.0f}")
    print(f"Median:  {lengths.median():,.0f}")
    print(f"75%:     {lengths.quantile(0.75):,.0f}")
    print(f"90%:     {lengths.quantile(0.90):,.0f}")
    print(f"95%:     {lengths.quantile(0.95):,.0f}")
    print(f"99%:     {lengths.quantile(0.99):,.0f}")
    print(f"Maximum: {lengths.max():,}")

    print("\nVery long messages:")

    for limit in [1000, 2000, 5000, 10000, 20000, 50000]:

        count = (lengths > limit).sum()

        percentage = (
            count / len(df) * 100
        )

        print(
            f"> {limit:>6,} chars: "
            f"{count:>7,} "
            f"({percentage:.2f}%)"
        )

    print("\nVery short messages:")

    for limit in [20, 50, 100, 200]:

        count = (lengths < limit).sum()

        percentage = (
            count / len(df) * 100
        )

        print(
            f"< {limit:>3} chars: "
            f"{count:>7,} "
            f"({percentage:.2f}%)"
        )

    print("\nClass × source:")

    table = pd.crosstab(
        df["source"],
        df["label"]
    )

    print(table.to_string())

    print("\nClass percentages:")

    class_percent = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(class_percent)


def main():

    print("=" * 80)
    print("TEXT DATASET DEEP PROFILE")
    print("=" * 80)

    for name, path in FILES.items():

        analyze(
            name,
            path
        )

    print("\n" + "=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()