import os
import json
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tqdm.auto import tqdm


MODEL_DIR = "ml/models/deberta_phishing/best"
TEST_FILE = "ml/data/processed/test_text.csv"

MAX_LENGTH = 256
BATCH_SIZE = 4


class EmailDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "label": self.labels[index]
        }


class Collator:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):

        texts = [
            item["text"]
            for item in batch
        ]

        labels = torch.tensor(
            [
                item["label"]
                for item in batch
            ],
            dtype=torch.long
        )

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded["labels"] = labels

        return encoded


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DEBERTA TEST SET EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Model: {MODEL_DIR}"
    )

    print(
        f"Test file: {TEST_FILE}"
    )

    print("=" * 70)

    print("\nLoading test dataset...")

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test samples: {len(test_df):,}"
    )

    print("\nTest labels:")

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    print("Tokenizer loaded.")

    dataset = EmailDataset(
        test_df
    )

    collator = Collator(
        tokenizer,
        MAX_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nLoading trained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    print(
        f"Model device: "
        f"{next(model.parameters()).device}"
    )

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("\nRunning test evaluation...")

    with torch.no_grad():

        progress = tqdm(
            dataloader,
            desc="Testing"
        )

        for batch in progress:

            labels = batch["labels"]

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                outputs = model(
                    **inputs
                )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .float()
                .cpu()
                .numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n")
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        "                 Predicted"
    )

    print(
        "                 Legit  Phishing"
    )

    print(
        f"Actual Legit     {cm[0][0]:6d}  {cm[0][1]:8d}"
    )

    print(
        f"Actual Phishing  {cm[1][0]:6d}  {cm[1][1]:8d}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "LEGITIMATE",
                "PHISHING"
            ],
            digits=4
        )
    )

    results = {
        "model": MODEL_DIR,
        "test_samples": len(test_df),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

    output_file = (
        "ml/models/deberta_phishing/"
        "test_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()