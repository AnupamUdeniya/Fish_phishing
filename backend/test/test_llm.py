from dotenv import load_dotenv
import os
from google.genai import types
from google import genai
import json
import re
import csv
import json
import random
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

correct = 0
errors = []

# Load your test dataset
with open('./Ling.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    emails = list(reader)


# Set a fixed seed so the random samples are the same every run
random.seed("1111122222")

# Separate by label
phishing_emails = [e for e in emails if e.get('label') == '1']
non_phishing_emails = [e for e in emails if e.get('label') == '0']

# Sample 5 from each
sampled_emails = random.sample(phishing_emails, 10) + random.sample(non_phishing_emails, 10)
random.shuffle(sampled_emails)

total = len(sampled_emails)

for i, email in enumerate(sampled_emails):
    chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
    max_output_tokens=200,
    )
    )
    
    
    body = email['body']
    subject = email['subject']
    prompt = f"""
    You are an expert in phishing email detection.

    Analyze the following email and respond in this JSON format:
    {{
    "is_phishing": true or false,
    "explanation": "A short explanation of why you classified it that way."
    }}

    Email:
    Subject: {subject}
    Body: {body}
    """

    try:
        response = chat.send_message(prompt)
        text = response.text.strip()

        # Clean code block markers if present
        if text.startswith("```json") or text.startswith("```"):
            text = text.strip("`").strip()
            text = text.split("\n", 1)[-1] if "\n" in text else text

        parsed = json.loads(text)

        predicted = 1 if parsed.get("is_phishing") else 0
        actual = int(email["label"])

        if predicted == actual:
            correct += 1
        else:
            errors.append((i, predicted, actual, email["subject"]))

    except Exception as e:
        errors.append((i, "Error", str(e)))
        
    time.sleep(0.25)#sleep for 1 sec

# Show results
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
print(f"Misclassified: {len(errors)}")
for e in errors[:5]:
    print(e)


    