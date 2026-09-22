import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


TRAIN_FILE = "ml/data/processed/train_url_domain.csv"
VAL_FILE = "ml/data/processed/val_url_domain.csv"
TEST_FILE = "ml/data/processed/test_url_domain.csv"

OUTPUT_DIR = "ml/data/processed/url_tfidf"

VECTORIZER_FILE = os.path.join(
    OUTPUT_DIR,
    "url_tfidf_vectorizer.joblib"
)

TRAIN_FILE_OUT = os.path.join(
    OUTPUT_DIR,
    "train_tfidf.npz"
)

VAL_FILE_OUT = os.path.join(
    OUTPUT_DIR,
    "val_tfidf.npz"
)

TEST_FILE_OUT = os.path.join(
    OUTPUT_DIR,
    "test_tfidf.npz"
)


def main():

    print("=" * 70)
    print("BUILDING CHARACTER-LEVEL URL TF-IDF")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load URLs
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["URL", "label"]
    )

    val = pd.read_csv(
        VAL_FILE,
        usecols=["URL", "label"]
    )

    test = pd.read_csv(
        TEST_FILE,
        usecols=["URL", "label"]
    )

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Validation: {len(val):,}"
    )

    print(
        f"Test: {len(test):,}"
    )

    # ---------------------------------------------------------
    # Convert URLs to strings
    # ---------------------------------------------------------

    train_urls = (
        train["URL"]
        .astype(str)
    )

    val_urls = (
        val["URL"]
        .astype(str)
    )

    test_urls = (
        test["URL"]
        .astype(str)
    )

    # ---------------------------------------------------------
    # TF-IDF vectorizer
    # ---------------------------------------------------------

    print("\nCreating character TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=100000,
        sublinear_tf=True,
        lowercase=True
    )

    # IMPORTANT:
    # Fit ONLY on training URLs.
    # Validation/test are transformed using
    # the training vocabulary.

    print("\nFitting vectorizer on training URLs...")

    X_train = vectorizer.fit_transform(
        train_urls
    )

    print(
        f"Train TF-IDF shape: {X_train.shape}"
    )

    print("\nTransforming validation URLs...")

    X_val = vectorizer.transform(
        val_urls
    )

    print(
        f"Validation TF-IDF shape: {X_val.shape}"
    )

    print("\nTransforming test URLs...")

    X_test = vectorizer.transform(
        test_urls
    )

    print(
        f"Test TF-IDF shape: {X_test.shape}"
    )

    # ---------------------------------------------------------
    # Convert to CSR
    # ---------------------------------------------------------

    X_train = csr_matrix(
        X_train
    )

    X_val = csr_matrix(
        X_val
    )

    X_test = csr_matrix(
        X_test
    )

    # ---------------------------------------------------------
    # Save matrices
    # ---------------------------------------------------------

    print("\nSaving TF-IDF matrices...")

    from scipy.sparse import save_npz

    save_npz(
        TRAIN_FILE_OUT,
        X_train
    )

    save_npz(
        VAL_FILE_OUT,
        X_val
    )

    save_npz(
        TEST_FILE_OUT,
        X_test
    )

    # ---------------------------------------------------------
    # Save labels
    # ---------------------------------------------------------

    train["label"].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train_labels.csv"
        ),
        index=False
    )

    val["label"].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "val_labels.csv"
        ),
        index=False
    )

    test["label"].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_labels.csv"
        ),
        index=False
    )

    # ---------------------------------------------------------
    # Save vectorizer
    # ---------------------------------------------------------

    print("\nSaving vectorizer...")

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TF-IDF RESULTS")
    print("=" * 70)

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Train matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Validation matrix: "
        f"{X_val.shape}"
    )

    print(
        f"Test matrix: "
        f"{X_test.shape}"
    )

    print(
        f"Train non-zero values: "
        f"{X_train.nnz:,}"
    )

    print(
        f"Validation non-zero values: "
        f"{X_val.nnz:,}"
    )

    print(
        f"Test non-zero values: "
        f"{X_test.nnz:,}"
    )

    print("\nSaved files:")

    print(
        VECTORIZER_FILE
    )

    print(
        TRAIN_FILE_OUT
    )

    print(
        VAL_FILE_OUT
    )

    print(
        TEST_FILE_OUT
    )

    print("\n" + "=" * 70)
    print("URL TF-IDF COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()