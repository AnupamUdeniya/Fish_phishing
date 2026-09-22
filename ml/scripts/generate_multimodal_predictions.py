import os
import re
import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier

from url_features import extract_url_features_dataframe


BASE_DIR = "ml"

DATA_DIR = f"{BASE_DIR}/data/processed"

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

OUTPUT_DIR = (
    f"{BASE_DIR}/data/processed/multimodal_predictions"
)

MAX_LENGTH = 256
TEXT_BATCH_SIZE = 4
URL_BATCH_SIZE = 512


def extract_urls(text):

    if not isinstance(text, str):
        return []

    urls = re.findall(
        r"https?://[^\s<>\[\]\"']+",
        text,
        flags=re.IGNORECASE
    )

    cleaned = []

    for url in urls:

        url = url.strip()

        url = url.rstrip(
            ".,;:!?)]}>"
        )

        if url:
            cleaned.append(url)

    return list(dict.fromkeys(cleaned))


class TextDataset(Dataset):

    def __init__(self, texts):

        self.texts = (
            texts.astype(str).tolist()
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def text_collator(batch, tokenizer):

    return tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


def predict_text(
    texts,
    tokenizer,
    model,
    device
):

    dataset = TextDataset(texts)

    loader = DataLoader(
        dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    probabilities = []

    total = len(dataset)

    print(
        f"Text samples: {total:,}"
    )

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            encoded = text_collator(
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

            processed = min(
                (batch_index + 1) * TEXT_BATCH_SIZE,
                total
            )

            if (
                batch_index == 0
                or processed % 1000 == 0
                or processed == total
            ):

                print(
                    f"  Text: "
                    f"{processed:,} / {total:,}"
                )

    return np.array(
        probabilities,
        dtype=np.float32
    )


def predict_urls(
    urls,
    url_model,
    tfidf,
    svd
):

    if len(urls) == 0:

        return np.array(
            [],
            dtype=np.float32
        )

    print(
        f"URLs to process: {len(urls):,}"
    )

    print("  Extracting handcrafted features...")

    handcrafted = (
        extract_url_features_dataframe(urls)
    )

    print(
        f"  Handcrafted: {handcrafted.shape}"
    )

    print("  Transforming URLs with TF-IDF...")

    tfidf_features = tfidf.transform(
        urls
    )

    print(
        f"  TF-IDF: {tfidf_features.shape}"
    )

    print("  Transforming with SVD...")

    svd_features = svd.transform(
        tfidf_features
    )

    print(
        f"  SVD: {svd_features.shape}"
    )

    handcrafted_array = (
        handcrafted.to_numpy(
            dtype=np.float32
        )
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
        f"  Combined: {combined.shape}"
    )

    probabilities = []

    total = len(combined)

    print("  Running XGBoost...")

    for start in range(
        0,
        total,
        URL_BATCH_SIZE
    ):

        end = min(
            start + URL_BATCH_SIZE,
            total
        )

        batch = combined[start:end]

        probs = (
            url_model
            .predict_proba(batch)[:, 1]
        )

        probabilities.extend(
            probs
        )

        print(
            f"  URLs: {end:,} / {total:,}"
        )

    return np.array(
        probabilities,
        dtype=np.float32
    )


def process_split(
    split_name,
    tokenizer,
    deberta,
    url_model,
    tfidf,
    svd,
    device
):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()} SET"
    )
    print("=" * 70)

    input_file = (
        f"{DATA_DIR}/{split_name}_text.csv"
    )

    output_file = (
        f"{OUTPUT_DIR}/{split_name}_multimodal.csv"
    )

    print(
        f"Loading: {input_file}"
    )

    df = pd.read_csv(
        input_file
    )

    print(
        f"Samples: {len(df):,}"
    )

    print("\nGenerating DeBERTa predictions...")

    deberta_probabilities = predict_text(
        df["text"],
        tokenizer,
        deberta,
        device
    )

    print("\nExtracting URLs...")

    email_urls = []

    url_counts = []

    for text in df["text"]:

        urls = extract_urls(text)

        email_urls.append(
            urls
        )

        url_counts.append(
            len(urls)
        )

    total_urls = sum(
        url_counts
    )

    emails_with_urls = sum(
        count > 0
        for count in url_counts
    )

    print(
        f"Emails with URLs: "
        f"{emails_with_urls:,} / {len(df):,}"
    )

    print(
        f"Total URLs: "
        f"{total_urls:,}"
    )

    flat_urls = [
        url
        for urls in email_urls
        for url in urls
    ]

    print("\nGenerating XGBoost URL predictions...")

    flat_url_probabilities = predict_urls(
        flat_urls,
        url_model,
        tfidf,
        svd
    )

    print("\nAggregating URL probabilities...")

    url_probabilities = []

    position = 0

    for urls in email_urls:

        if len(urls) == 0:

            url_probabilities.append(
                0.0
            )

        else:

            count = len(urls)

            current_probs = (
                flat_url_probabilities[
                    position:
                    position + count
                ]
            )

            url_probabilities.append(
                float(
                    np.max(
                        current_probs
                    )
                )
            )

            position += count

    result = pd.DataFrame(
        {
            "id": df["id"],
            "source": df["source"],
            "label": df["label"],
            "deberta_probability":
                deberta_probabilities,
            "url_probability":
                url_probabilities,
            "url_count":
                url_counts
        }
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result.to_csv(
        output_file,
        index=False
    )

    print("\nSplit complete.")

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print("\nPrediction statistics:")

    print(
        result[
            [
                "deberta_probability",
                "url_probability",
                "url_count"
            ]
        ].describe()
    )


def main():

    print("=" * 70)
    print("LARGE-SCALE MULTIMODAL PREDICTION GENERATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nLoading DeBERTa...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            DEBERTA_DIR
        )
    )

    deberta = (
        AutoModelForSequenceClassification
        .from_pretrained(DEBERTA_DIR)
    )

    deberta.to(device)
    deberta.eval()

    print("DeBERTa loaded.")

    print("\nLoading XGBoost...")

    url_model = XGBClassifier()

    url_model.load_model(
        XGB_MODEL_FILE
    )

    print("XGBoost loaded.")

    print("\nLoading TF-IDF vectorizer...")

    tfidf = joblib.load(
        TFIDF_FILE
    )

    print("TF-IDF loaded.")

    print("\nLoading SVD model...")

    svd = joblib.load(
        SVD_FILE
    )

    print("SVD loaded.")

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        process_split(
            split_name,
            tokenizer,
            deberta,
            url_model,
            tfidf,
            svd,
            device
        )

    print("\n")
    print("=" * 70)
    print("ALL MULTIMODAL PREDICTIONS COMPLETE")
    print("=" * 70)

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  train_multimodal.csv"
    )

    print(
        "  val_multimodal.csv"
    )

    print(
        "  test_multimodal.csv"
    )


if __name__ == "__main__":
    main()