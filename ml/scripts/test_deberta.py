import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "microsoft/deberta-v3-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\n===== DEVICE TEST =====")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)

texts = [
    "Your account has been successfully updated. Thank you for using our service.",
    "URGENT! Your account will be suspended. Click this link immediately to verify your password."
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

inputs = {key: value.to(device) for key, value in inputs.items()}

print("\nRunning DeBERTa on GPU...")

with torch.no_grad():
    outputs = model(**inputs)

print("Output shape:", outputs.logits.shape)
print("Logits:")
print(outputs.logits)

print("\n===== SUCCESS =====")
print("DeBERTa is running on your RTX 4060.")
