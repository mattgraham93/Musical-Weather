import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Musical Weather App is running!"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "port": os.environ.get('PORT', '5000')
    })

if __name__ == "__main__":
    print("Starting minimal Flask app...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)