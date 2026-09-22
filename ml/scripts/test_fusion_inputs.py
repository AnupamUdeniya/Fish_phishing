import re
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


TEXT_MODEL = "ml/models/deberta_phishing/best"
URL_MODEL = "ml/models/xgboost_url/url_xgboost.json"
FEATURE_FILE = "ml/models/xgboost_url/features.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_LENGTH = 256


def extract_urls(text):

    pattern = r"https?://[^\s<>\"]+"

    urls = re.findall(
        pattern,
        text
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,!?;:)]}"
        )

        cleaned.append(url)

    return cleaned


def get_text_probability(text, tokenizer, model):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            outputs = model(
                **inputs
            )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    return float(
        probabilities[0, 1]
        .float()
        .cpu()
        .item()
    )


def create_url_features(url):

    from urllib.parse import urlparse

    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    letters = sum(
        c.isalpha()
        for c in url
    )

    digits = sum(
        c.isdigit()
        for c in url
    )

    special = sum(
        not c.isalnum()
        for c in url
    )

    subdomains = max(
        len(domain.split(".")) - 2,
        0
    )

    is_ip = int(
        bool(
            re.fullmatch(
                r"\d{1,3}(\.\d{1,3}){3}",
                domain.split(":")[0]
            )
        )
    )

    return {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "NoOfSubDomain": subdomains,
        "HasObfuscation": int(
            "@" in url or
            "%" in url
        ),
        "NoOfObfuscatedChar": (
            url.count("%")
        ),
        "ObfuscationRatio": (
            url.count("%") / max(len(url), 1)
        ),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": (
            letters / max(len(url), 1)
        ),
        "NoOfDegitsInURL": digits,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL": (
            special / max(len(url), 1)
        ),
        "IsHTTPS": int(
            url.lower().startswith("https://")
        ),
        "NoOfURLRedirect": (
            url.count("http://") +
            url.count("https://") -
            1
        ),
        "NoOfSelfRedirect": 0,
        "HasExternalFormSubmit": 0,
        "HasSocialNet": int(
            any(
                x in url.lower()
                for x in [
                    "facebook",
                    "instagram",
                    "twitter",
                    "linkedin",
                    "youtube",
                    "tiktok"
                ]
            )
        ),
        "HasPasswordField": int(
            "password" in url.lower()
        ),
        "Bank": int(
            "bank" in url.lower()
        ),
        "Pay": int(
            any(
                x in url.lower()
                for x in [
                    "pay",
                    "payment",
                    "paypal"
                ]
            )
        ),
        "Crypto": int(
            any(
                x in url.lower()
                for x in [
                    "crypto",
                    "bitcoin",
                    "wallet"
                ]
            )
        ),
        "NoOfExternalRef": 0
    }


def main():

    print("=" * 70)
    print("FUSION INPUT TEST")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("\nLoading DeBERTa...")

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL
    )

    text_model = (
        AutoModelForSequenceClassification
        .from_pretrained(TEXT_MODEL)
    )

    text_model.to(DEVICE)
    text_model.eval()

    print("DeBERTa loaded.")

    print("\nLoading XGBoost...")

    url_model = xgb.XGBClassifier()

    url_model.load_model(
        URL_MODEL
    )

    print("XGBoost loaded.")

    print("\nTesting sample email...")

    sample = """
    Your account requires immediate verification.
    Please confirm your information by visiting
    https://secure-example-login.com/verify
    """

    print("\nSample:")
    print(sample)

    text_probability = get_text_probability(
        sample,
        tokenizer,
        text_model
    )

    urls = extract_urls(
        sample
    )

    print(
        f"\nURLs found: {urls}"
    )

    url_probability = 0.0

    if urls:

        url = urls[0]

        features = create_url_features(
            url
        )

        feature_df = pd.DataFrame(
            [features]
        )

        feature_names = [
            "URLLength",
            "DomainLength",
            "IsDomainIP",
            "NoOfSubDomain",
            "HasObfuscation",
            "NoOfObfuscatedChar",
            "ObfuscationRatio",
            "NoOfLettersInURL",
            "LetterRatioInURL",
            "NoOfDegitsInURL",
            "NoOfEqualsInURL",
            "NoOfQMarkInURL",
            "NoOfAmpersandInURL",
            "NoOfOtherSpecialCharsInURL",
            "SpacialCharRatioInURL",
            "IsHTTPS",
            "NoOfURLRedirect",
            "NoOfSelfRedirect",
            "HasExternalFormSubmit",
            "HasSocialNet",
            "HasPasswordField",
            "Bank",
            "Pay",
            "Crypto",
            "NoOfExternalRef"
        ]

        feature_df = feature_df[
            feature_names
        ]

        url_probability = float(
            url_model
            .predict_proba(feature_df)[0, 1]
        )

    print("\n" + "=" * 70)
    print("MODEL OUTPUTS")
    print("=" * 70)

    print(
        f"DeBERTa phishing probability: "
        f"{text_probability:.4f}"
    )

    print(
        f"XGBoost URL probability:       "
        f"{url_probability:.4f}"
    )

    fusion_probability = (
        0.6 * text_probability +
        0.4 * url_probability
    )

    print(
        f"Simple fusion probability:     "
        f"{fusion_probability:.4f}"
    )

    if fusion_probability >= 0.5:

        print(
            "\nFINAL: PHISHING"
        )

    else:

        print(
            "\nFINAL: LEGITIMATE"
        )

    print("\n" + "=" * 70)
    print("FUSION INPUT TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()