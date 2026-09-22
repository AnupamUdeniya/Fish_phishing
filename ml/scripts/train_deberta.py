import os
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from tqdm.auto import tqdm


MODEL_NAME = "microsoft/deberta-v3-base"

TRAIN_FILE = "ml/data/processed/train_text.csv"
VAL_FILE = "ml/data/processed/val_text.csv"

OUTPUT_DIR = "ml/models/deberta_phishing"

MAX_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 2

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        texts = [item["text"] for item in batch]

        labels = torch.tensor(
            [item["label"] for item in batch],
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


def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        progress = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for batch in progress:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                outputs = model(**batch)

            loss = outputs.loss

            predictions = torch.argmax(
                outputs.logits,
                dim=-1
            )

            total_loss += loss.item()
            total_batches += 1

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    avg_loss = total_loss / max(total_batches, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels
    }


def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("DEBERTA PHISHING DETECTION TRAINING")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("=" * 70)

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    print(
        f"Training samples: {len(train_df):,}"
    )

    print(
        f"Validation samples: {len(val_df):,}"
    )

    print("\nTraining labels:")
    print(
        train_df["label"].value_counts().sort_index()
    )

    print("\nValidation labels:")
    print(
        val_df["label"].value_counts().sort_index()
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    train_dataset = EmailDataset(train_df)
    val_dataset = EmailDataset(val_df)

    collator = Collator(
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nLoading DeBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "LEGITIMATE",
            1: "PHISHING"
        },
        label2id={
            "LEGITIMATE": 0,
            "PHISHING": 1
        }
    )

    model.config.problem_type = "single_label_classification"

    model = model.float()

    if hasattr(model, "gradient_checkpointing_enable"):

        model.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: ENABLED"
        )

    model.to(device)

    print(
        f"Model device: "
        f"{next(model.parameters()).device}"
    )

    print(
        f"Model dtype: "
        f"{next(model.parameters()).dtype}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_update_steps = (
        len(train_loader)
        // GRADIENT_ACCUMULATION_STEPS
    )

    if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:

        total_update_steps += 1

    total_training_steps = (
        total_update_steps * EPOCHS
    )

    warmup_steps = int(
        total_training_steps * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    print("\nTraining configuration:")

    print(
        f"Training batches per epoch: "
        f"{len(train_loader):,}"
    )

    print(
        f"Optimizer updates per epoch: "
        f"{total_update_steps:,}"
    )

    print(
        f"Total optimizer updates: "
        f"{total_training_steps:,}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps:,}"
    )

    best_f1 = -1.0

    training_history = []

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}",
            leave=True
        )

        for step, batch in enumerate(progress):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                outputs = model(
                    **batch
                )

                loss = outputs.loss

                loss = (
                    loss /
                    GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(loss).backward()

            running_loss += loss.item()

            should_update = (
                (step + 1)
                % GRADIENT_ACCUMULATION_STEPS == 0
                or
                (step + 1) == len(train_loader)
            )

            if should_update:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            current_loss = (
                running_loss /
                (step + 1)
            )

            progress.set_postfix(
                loss=f"{current_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        print("\nRunning validation...")

        metrics = evaluate(
            model,
            val_loader,
            device
        )

        print("\nValidation Results")
        print("-" * 50)

        print(
            f"Loss:      {metrics['loss']:.4f}"
        )

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:  {metrics['f1']:.4f}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "validation_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            best_dir = os.path.join(
                OUTPUT_DIR,
                "best"
            )

            os.makedirs(
                best_dir,
                exist_ok=True
            )

            print(
                f"\nNew best F1: {best_f1:.4f}"
            )

            print(
                "Saving best model..."
            )

            model.save_pretrained(
                best_dir
            )

            tokenizer.save_pretrained(
                best_dir
            )

            with open(
                os.path.join(
                    best_dir,
                    "metrics.json"
                ),
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    training_history,
                    f,
                    indent=4
                )

            print(
                f"Best model saved to: "
                f"{best_dir}"
            )

        if device.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            print(
                f"GPU memory: "
                f"{allocated:.2f} GB allocated / "
                f"{reserved:.2f} GB reserved"
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation F1: {best_f1:.4f}"
    )

    print(
        f"Best model location: "
        f"{OUTPUT_DIR}\\best"
    )

    final_metrics = evaluate(
        model,
        val_loader,
        device
    )

    print("\nFinal Classification Report")
    print("=" * 70)

    print(
        classification_report(
            final_metrics["labels"],
            final_metrics["predictions"],
            target_names=[
                "LEGITIMATE",
                "PHISHING"
            ],
            digits=4
        )
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "training_history.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            training_history,
            f,
            indent=4
        )

    print(
        f"Training history saved to: "
        f"{OUTPUT_DIR}\\training_history.json"
    )


if __name__ == "__main__":
    main()