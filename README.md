# Fish Phishing Detection

A browser-extension phishing detection system for Gmail that checks email text and URLs, combines the scores, and warns the user when a message looks suspicious.

## Features
- Gmail email extraction
- Local Flask backend
- URL and text analysis
- Risk score and detection result
- Chrome extension integration

## Project structure
- backend/ - Flask API and model loading
- extension/ - Chrome extension files
- frontend/ - browser popup UI
- ml/ - datasets, trained models and training scripts

## Run the project

### Backend
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Frontend
```powershell
cd frontend/phishing-extension
npm install
npm run build
```

### Load in Chrome
1. Open chrome://extensions
2. Enable Developer mode
3. Click Load unpacked
4. Select the extension folder
5. Refresh Gmail and test a message

## Notes
This project is designed for local phishing detection and does not depend on an external service during inference.
