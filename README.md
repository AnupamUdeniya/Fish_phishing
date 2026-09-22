# Fish Phishing Detection

An AI-powered phishing detection system that combines **Deep Learning, Machine Learning, URL analysis, and browser-extension technology** to detect phishing emails and malicious URLs.

The project started from an existing LLM-based browser extension and is being upgraded into a system with **independently trained ML/DL models**, reducing dependency on external LLM APIs.

---

# 📌 Project Overview

Phishing attacks commonly use fraudulent emails, malicious URLs, fake login pages, urgency-based messages, and social engineering techniques to deceive users.

The original version of this project used an external Large Language Model (LLM) API to classify emails.

This upgraded version focuses on building our own machine-learning pipeline using:

- **DeBERTa-v3-base** for email/text phishing detection
- **XGBoost** for URL phishing detection
- URL handcrafted features
- Character-level URL TF-IDF
- Truncated SVD
- Domain-aware dataset splitting
- Late-fusion experiments combining text and URL predictions
- Chrome browser extension integration

The long-term goal is to create a phishing detection system that can analyze:

1. Email text
2. URLs
3. Multiple URLs inside an email
4. Combined text + URL evidence

---

# 🎯 Project Goals

The main goals of the project are:

- Detect phishing emails using a self-trained Deep Learning model.
- Detect malicious/phishing URLs using a Machine Learning model.
- Analyze both text and URLs rather than relying only on one signal.
- Reduce dependency on external LLM APIs.
- Build a model that can eventually run as part of a browser extension.
- Provide phishing probability/risk information.
- Compare individual models with a combined multimodal approach.
- Create a reproducible ML pipeline for dataset preparation, training, and evaluation.

---

# 👥 How People Use This Project

## Students and Researchers

Students can use this project to learn how a complete phishing detection system is developed. It demonstrates dataset preparation, text classification, URL feature extraction, model evaluation, Flask API development, and browser-extension integration in one project.

Researchers can use the training scripts and evaluation files to compare text-based detection, URL-based detection, and combined text-plus-URL detection approaches.

## Gmail Users

Gmail users can load the Chrome extension locally and scan an opened email before clicking links or sharing sensitive information. The extension extracts the email subject, sender, body, and available URLs, then displays a phishing result and risk information in the popup.

## Developers

Developers can use the Flask API as a local service for testing phishing detection workflows. The frontend can also be modified to support additional email providers, new UI features, or improved model responses.

## Security Demonstrations

The project can be used in classroom demonstrations, project presentations, and controlled security-awareness exercises. A demonstrator can show how urgent language, suspicious login requests, and unsafe-looking URLs affect the final classification.

## Email Testing Workflow

1. Start the local Flask backend.
2. Build and load the Chrome extension.
3. Open Gmail and select an email.
4. Open the Phishing Detector popup.
5. Review the extracted subject, sender, and message preview.
6. Select **Scan this email**.
7. Review the classification, risk score, and detected URLs.

The project should only be used with emails and datasets that the user is authorized to inspect. It is intended for education, research, and controlled testing, not as a replacement for enterprise email security systems.

---

# 🏗️ System Architecture

## Current Architecture

```text
                         EMAIL / TEXT
                              |
                              v
                    +-------------------+
                    |    DeBERTa-v3     |
                    |      Base         |
                    +-------------------+
                              |
                              v
                    Text Phishing Score
                              |
                              |
                              |
                              v
                       +-------------+
                       |   Fusion    |
                       |    Layer    |
                       +-------------+
                              ^
                              |
                              |
                       URL Phishing Score
                              ^
                              |
                    +-------------------+
                    |     XGBoost       |
                    |   URL Detector   |
                    +-------------------+
                              ^
                              |
                    +-------------------+
                    | URL Representation|
                    +-------------------+
                       /             \
                      /               \
                     v                 v
          Handcrafted Features      URL TF-IDF
                                      |
                                      v
                                  SVD (300)
