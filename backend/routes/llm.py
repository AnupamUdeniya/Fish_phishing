from dotenv import load_dotenv
import os
import re
import sys
from pathlib import Path
from flask import Blueprint, jsonify, request
import json

llm_bp = Blueprint('llm', __name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEXT_MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "deberta_phishing" / "best"
URL_MODEL_FILE = PROJECT_ROOT / "ml" / "models" / "xgboost_url" / "url_xgboost.json"
MODEL_STATE = None


def load_models():
    global MODEL_STATE

    if MODEL_STATE is not None:
        return MODEL_STATE

    try:
        import pandas as pd
        import torch
        import xgboost as xgb
        from transformers import AutoModelForSequenceClassification, DebertaV2TokenizerFast
        from ml.scripts.url_features import extract_url_features_dataframe
    except ImportError as error:
        raise RuntimeError(
            "Trained-model dependencies are missing. Run "
            "pip install -r backend/requirements.txt."
        ) from error

    if not TEXT_MODEL_DIR.exists() or not URL_MODEL_FILE.exists():
        raise RuntimeError("Trained model artifacts are missing from ml/models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DebertaV2TokenizerFast(
        tokenizer_file=str(TEXT_MODEL_DIR / "tokenizer.json"),
        bos_token="[CLS]",
        eos_token="[SEP]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    text_model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_MODEL_DIR))
    text_model.to(device)
    text_model.eval()

    url_model = xgb.XGBClassifier()
    url_model.load_model(str(URL_MODEL_FILE))

    MODEL_STATE = {
        "device": device,
        "torch": torch,
        "tokenizer": tokenizer,
        "text_model": text_model,
        "url_model": url_model,
        "extract_url_features_dataframe": extract_url_features_dataframe,
    }
    return MODEL_STATE


def extract_urls(text):
    urls = re.findall(r"https?://[^\s<>\"]+", text or "")
    return [url.rstrip(".,!?;:)]}") for url in urls]


def predict_text(text, state):
    inputs = state["tokenizer"](
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs = {key: value.to(state["device"]) for key, value in inputs.items()}
    with state["torch"].no_grad():
        logits = state["text_model"](**inputs).logits
    return float(state["torch"].softmax(logits, dim=-1)[0, 1].cpu().item())


def predict_url(urls, state):
    if not urls:
        return 0.0
    features = state["extract_url_features_dataframe"](urls)
    probabilities = state["url_model"].predict_proba(features)[:, 1]
    return float(max(probabilities))


@llm_bp.route('/api/ml-detect', methods=['POST'])
def ml_detect():
    data = request.get_json(silent=True) or {}
    subject = str(data.get("subject", ""))
    sender = str(data.get("sender", ""))
    body = str(data.get("body", ""))
    message = f"Subject: {subject}\nFrom: {sender}\n\n{body}"

    try:
        state = load_models()
        urls = extract_urls(message)
        text_probability = predict_text(message, state)
        url_probability = predict_url(urls, state)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    risk_probability = 0.6 * text_probability + 0.4 * url_probability
    is_phishing = risk_probability >= 0.5
    return jsonify({
        "is_phishing": is_phishing,
        "risk_level": "High" if risk_probability >= 0.7 else "Medium" if is_phishing else "Low",
        "risk_score": round(risk_probability * 100, 2),
        "text_probability": round(text_probability * 100, 2),
        "url_probability": round(url_probability * 100, 2),
        "url_count": len(urls),
        "urls": urls
    })


@llm_bp.route('/api/llm-query', methods=['POST'])
def generate():
    data = request.json
    subject = data.get("subject")
    sender = data.get("sender")
    body = data.get("body")
    
    prompt = f"""
    You are an expert in phishing email detection.

    Analyze the following email and respond in this JSON format:
    {{
    "is_phishing": true or false,
    "explanation": "A short explanation of why you classified it that way."
    }}

    Email:
    Subject: {subject}
    From: {sender}
    Body: {body}
    """
    
    if not prompt:
        return jsonify({"error": "Missing message"}), 400
    
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY is not configured."}), 503

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(max_output_tokens=200)
        )
        response = chat.send_message(prompt)
    except Exception:
        return jsonify({"error": "Gemini request failed."}), 502

    print(response.text, flush=True)
    
    response_text = response.text.strip()
    
    # Remove Markdown code block markers if present
    if response_text.startswith("```json") or response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip())

    try:
        # Try to parse JSON output from the model
        parsed = json.loads(response_text)
        return jsonify({
            "is_phishing": parsed.get("is_phishing"),
            "explanation": parsed.get("explanation")
        })
    except Exception as e:
        # Fallback in case the model output is not valid JSON
        return jsonify({
            "error": "Failed to parse model response",
            "raw_response": response.text
        }), 500


# @llm_bp.route('/api-analyze', methods = ['POST'])
# def analyze():
#     data = request.get_json()
#     email_body = data.get("body","").lower()

#     if "click here" in email_body or "urgent" in email_body or "verify" in email_body:
#         label = "Phishing"
#         explanation = "Contains urgent language and call-to-action."
#     elif "please review" in email_body or "unknown sender" in email_body:
#         label = "Suspicious"
#         explanation = "Might be legitimate but has warning signs."
#     else:
#         label = "Safe"
#         explanation = "No phishing signals detected."

#     return jsonify({
#         "label": label,
#         "explanation": explanation
#     })