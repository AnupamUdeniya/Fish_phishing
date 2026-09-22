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
