import os
import pandas as pd
from tqdm import tqdm

from url_features import (
    extract_url_features,
    URL_FEATURES
)


INPUT_FILES = {
    "train": "ml/data/processed/train_url_domain.csv",
    "val": "ml/data/processed/val_url_domain.csv",
    "test": "ml/data/processed/test_url_domain.csv",
}

OUTPUT_DIR = "ml/data/processed"


def process_split(name, input_file):

    print("\n" + "=" * 70)
    print(f"PROCESSING {name.upper()} SET")
    print("=" * 70)

    df = pd.read_csv(
        input_file
    )

    print(
        f"Input samples: {len(df):,}"
    )

    feature_rows = []

    for url in tqdm(
        df["URL"].astype(str),
        total=len(df),
        desc=f"{name} URLs"
    ):

        feature_rows.append(
            extract_url_features(url)
        )

    features_df = pd.DataFrame(
        feature_rows,
        columns=URL_FEATURES
    )

    features_df["label"] = (
        df["label"]
        .astype(int)
        .values
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_url_features.csv"
    )

    features_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Output shape: {features_df.shape}"
    )

    print(
        f"Missing values: "
        f"{features_df.isna().sum().sum()}"
    )

    print(
        f"Saved: {output_file}"
    )


def main():

    print("=" * 70)
    print("BUILDING DOMAIN-SAFE URL FEATURES")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for name, input_file in INPUT_FILES.items():

        process_split(
            name,
            input_file
        )

    print("\n" + "=" * 70)
    print("DOMAIN-SAFE URL FEATURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()