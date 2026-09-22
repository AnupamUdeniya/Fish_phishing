from dotenv import load_dotenv
import os
from flask import Blueprint, jsonify, request
import json
import re

llm_bp = Blueprint('llm', __name__)


def local_detection(subject, sender, body):
    message = f"{subject} {sender} {body}".lower()
    indicators = (
        "urgent", "verify your account", "click here", "password",
        "suspended", "confirm your", "login", "winner", "gift card"
    )
    matches = [indicator for indicator in indicators if indicator in message]

    if matches:
        return jsonify({
            "is_phishing": True,
            "explanation": "Local analysis found phishing indicators: " + ", ".join(matches) + "."
        })

    return jsonify({
        "is_phishing": False,
        "explanation": "Local analysis did not find strong phishing indicators."
    })


@llm_bp.route('/api/ml-detect', methods=['POST'])
def ml_detect():
    data = request.get_json(silent=True) or {}
    result = local_detection(
        data.get("subject", ""),
        data.get("sender", ""),
        data.get("body", "")
    ).get_json()
    is_phishing = bool(result.get("is_phishing"))
    return jsonify({
        "is_phishing": is_phishing,
        "risk_level": "High" if is_phishing else "Low",
        "risk_score": 85 if is_phishing else 10,
        "text_probability": 85 if is_phishing else 10,
        "url_probability": 0,
        "url_count": 0,
        "urls": []
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
        return local_detection(subject, sender, body)

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
        return local_detection(subject, sender, body)

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