# Fish Phishing Detection

An AI-powered phishing detection browser extension that combines
Deep Learning and Machine Learning models to detect phishing emails
and malicious URLs.

## Project Overview

This project is an upgraded version of an existing phishing detection
browser extension.

The original system used an LLM-based approach for phishing detection.
Our version is being extended with independently trained Machine
Learning and Deep Learning models.

## Current Architecture

```text
Email / Text
     |
     v
DeBERTa-v3-base
     |
     v
Text Phishing Probability


URL
 |
 v
URL Feature Extraction
 |
 +--> Handcrafted Features
 |
 +--> URL TF-IDF
 |
 +--> SVD
 |
 v
XGBoost
 |
 v
URL Phishing Probability

        |
        v
   Fusion Layer
        |
        v
Final Phishing Detection
