import os
import re
import ast
import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier

from url_features import extract_url_features_dataframe


BASE_DIR = "ml"

SPAPHISH_FILE = f"{BASE_DIR}/data/processed/spaphish_dataset.csv"

DEBERTA_DIR = f"{BASE_DIR}/models/deberta_phishing/best"

XGB_MODEL_FILE = (
    f"{BASE_DIR}/models/xgboost_url/"
    "url_xgboost_improved.json"
)

TFIDF_FILE = (
    f"{BASE_DIR}/data/processed/url_tfidf/"
    "url_tfidf_vectorizer.joblib"
)

SVD_FILE = (
    f"{BASE_DIR}/data/processed/url_svd/"
    "url_svd.joblib"
)

OUTPUT_FILE = (
    f"{BASE_DIR}/data/processed/"
    "fusion_predictions.csv"
)

MAX_LENGTH = 256
TEXT_BATCH_SIZE = 4
URL_BATCH_SIZE = 512


class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts.astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def collate_texts(batch, tokenizer):

    return tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


def extract_urls(value):

    if not isinstance(value, str):
        return []

    urls = re.findall(
        r"https?://[^\s\],]+",
        value,
        flags=re.IGNORECASE
    )

    cleaned = []

    for url in urls:

        url = url.strip()

        url = url.rstrip(
            ".,;:!?)]}'\""
        )

        if url:
            cleaned.append(url)

    return list(dict.fromkeys(cleaned))


def load_deberta(device):

    print("\nLoading DeBERTa...")

    tokenizer = AutoTokenizer.from_pretrained(
        DEBERTA_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_DIR
    )

    model.to(device)
    model.eval()

    print("DeBERTa loaded.")

    return tokenizer, model


def predict_text(
    texts,
    tokenizer,
    model,
    device
):

    dataset = TextDataset(texts)

    dataloader = DataLoader(
        dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    probabilities = []

    print("\nRunning DeBERTa predictions...")

    with torch.no_grad():

        for batch in dataloader:

            encoded = collate_texts(
                batch,
                tokenizer
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                outputs = model(
                    **encoded
                )

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            probabilities.extend(
                probs[:, 1]
                .float()
                .cpu()
                .numpy()
            )

    return np.array(probabilities)


def predict_urls(
    urls,
    url_model,
    tfidf,
    svd,
    device
):

    print("\nExtracting URL features...")

    handcrafted = extract_url_features_dataframe(
        urls
    )

    print(
        f"Handcrafted features: "
        f"{handcrafted.shape}"
    )

    print("\nGenerating TF-IDF features...")

    tfidf_features = tfidf.transform(
        urls
    )

    print(
        f"TF-IDF shape: "
        f"{tfidf_features.shape}"
    )

    print("\nGenerating SVD features...")

    svd_features = svd.transform(
        tfidf_features
    )

    print(
        f"SVD shape: "
        f"{svd_features.shape}"
    )

    handcrafted_array = handcrafted.to_numpy(
        dtype=np.float32
    )

    svd_array = np.asarray(
        svd_features,
        dtype=np.float32
    )

    combined = np.hstack(
        [
            handcrafted_array,
            svd_array
        ]
    )

    print(
        f"Combined URL features: "
        f"{combined.shape}"
    )

    print("\nRunning XGBoost predictions...")

    probabilities = []

    for start in range(
        0,
        len(combined),
        URL_BATCH_SIZE
    ):

        end = min(
            start + URL_BATCH_SIZE,
            len(combined)
        )

        batch = combined[start:end]

        probs = url_model.predict_proba(
            batch
        )[:, 1]

        probabilities.extend(
            probs
        )

        print(
            f"Processed URLs: "
            f"{end:,} / {len(combined):,}"
        )

    return np.array(probabilities)


def main():

    print("=" * 70)
    print("GENERATING FUSION PREDICTIONS")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nLoading SpaPhish dataset...")

    df = pd.read_csv(
        SPAPHISH_FILE
    )

    print(
        f"Samples: {len(df):,}"
    )

    print("\nLabels:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nExtracting URLs from SpaPhish...")

    all_urls = []

    email_url_counts = []

    for value in df["url"]:

        urls = extract_urls(value)

        all_urls.append(urls)

        email_url_counts.append(
            len(urls)
        )

    total_urls = sum(
        email_url_counts
    )

    emails_with_urls = sum(
        count > 0
        for count in email_url_counts
    )

    print(
        f"Emails containing URLs: "
        f"{emails_with_urls:,} / {len(df):,}"
    )

    print(
        f"Total extracted URLs: "
        f"{total_urls:,}"
    )

    flat_urls = [
        url
        for urls in all_urls
        for url in urls
    ]

    if len(flat_urls) == 0:

        raise RuntimeError(
            "No URLs were extracted from SpaPhish."
        )

    print("\nLoading URL models...")

    print("Loading XGBoost model...")

    url_model = XGBClassifier()

    url_model.load_model(
        XGB_MODEL_FILE
    )

    print("XGBoost loaded.")

    print("Loading TF-IDF vectorizer...")

    tfidf = joblib.load(
        TFIDF_FILE
    )

    print("TF-IDF loaded.")

    print("Loading SVD model...")

    svd = joblib.load(
        SVD_FILE
    )

    print("SVD loaded.")

    tokenizer, deberta = load_deberta(
        device
    )

    text_probabilities = predict_text(
        df["text"],
        tokenizer,
        deberta,
        device
    )

    url_probabilities_flat = predict_urls(
        flat_urls,
        url_model,
        tfidf,
        svd,
        device
    )

    print("\nAggregating URL probabilities...")

    email_url_probabilities = []

    position = 0

    for urls in all_urls:

        if len(urls) == 0:

            email_url_probabilities.append(
                0.0
            )

        else:

            count = len(urls)

            email_probs = (
                url_probabilities_flat[
                    position:
                    position + count
                ]
            )

            email_url_probabilities.append(
                float(
                    np.max(
                        email_probs
                    )
                )
            )

            position += count

    result = pd.DataFrame(
        {
            "id": df["id"],
            "label": df["label"],
            "deberta_probability":
                text_probabilities,
            "url_probability":
                email_url_probabilities,
            "url_count":
                email_url_counts
        }
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FUSION PREDICTIONS COMPLETE")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print("\nPreview:")

    print(
        result.head(10).to_string(
            index=False
        )
    )

    print("\nProbability statistics:")

    print(
        result[
            [
                "deberta_probability",
                "url_probability"
            ]
        ].describe()
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()