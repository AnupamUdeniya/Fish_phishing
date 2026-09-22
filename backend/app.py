from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from routes.llm import llm_bp

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.register_blueprint(llm_bp)

CORS(app)


@app.get('/api/health')
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False)