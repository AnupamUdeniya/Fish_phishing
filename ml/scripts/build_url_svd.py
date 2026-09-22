import os
import joblib
import numpy as np

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD


INPUT_DIR = "ml/data/processed/url_tfidf"
OUTPUT_DIR = "ml/data/processed/url_svd"

N_COMPONENTS = 300


def main():

    print("=" * 70)
    print("BUILDING URL TF-IDF SVD REPRESENTATION")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading TF-IDF matrices...")

    X_train = load_npz(
        os.path.join(INPUT_DIR, "train_tfidf.npz")
    )

    X_val = load_npz(
        os.path.join(INPUT_DIR, "val_tfidf.npz")
    )

    X_test = load_npz(
        os.path.join(INPUT_DIR, "test_tfidf.npz")
    )

    print(f"Train TF-IDF: {X_train.shape}")
    print(f"Validation TF-IDF: {X_val.shape}")
    print(f"Test TF-IDF: {X_test.shape}")

    print("\nCreating Truncated SVD...")
    print(f"Components: {N_COMPONENTS}")

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        algorithm="randomized",
        n_iter=5,
        random_state=42
    )

    print("\nFitting SVD on training data only...")

    X_train_svd = svd.fit_transform(X_train)

    print(
        f"Train SVD shape: {X_train_svd.shape}"
    )

    print("\nTransforming validation data...")

    X_val_svd = svd.transform(X_val)

    print(
        f"Validation SVD shape: {X_val_svd.shape}"
    )

    print("\nTransforming test data...")

    X_test_svd = svd.transform(X_test)

    print(
        f"Test SVD shape: {X_test_svd.shape}"
    )

    explained = svd.explained_variance_ratio_.sum()

    print("\n" + "=" * 70)
    print("SVD RESULTS")
    print("=" * 70)

    print(
        f"Explained variance: {explained:.4f}"
    )

    print(
        f"Explained variance: {explained * 100:.2f}%"
    )

    print(
        f"Original dimensions: {X_train.shape[1]:,}"
    )

    print(
        f"Reduced dimensions: {N_COMPONENTS}"
    )

    print("\nSaving SVD features...")

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "train_svd.npy"
        ),
        X_train_svd.astype(np.float32)
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "val_svd.npy"
        ),
        X_val_svd.astype(np.float32)
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "test_svd.npy"
        ),
        X_test_svd.astype(np.float32)
    )

    print("\nSaving SVD model...")

    joblib.dump(
        svd,
        os.path.join(
            OUTPUT_DIR,
            "url_svd.joblib"
        )
    )

    print("\nFiles saved:")

    print(
        "ml/data/processed/url_svd/train_svd.npy"
    )

    print(
        "ml/data/processed/url_svd/val_svd.npy"
    )

    print(
        "ml/data/processed/url_svd/test_svd.npy"
    )

    print(
        "ml/data/processed/url_svd/url_svd.joblib"
    )

    print("\n" + "=" * 70)
    print("URL SVD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()