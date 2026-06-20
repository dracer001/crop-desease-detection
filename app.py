"""
FarmBot Backend Server
=======================
Flask backend that receives a crop-leaf image from an ESP32-CAM device,
runs it through the trained Keras model (cassava_model.keras), and returns
a structured diagnosis (status, severity, confidence, advice) as JSON.

SETUP
-----
1. Place your trained model file next to this script (or set MODEL_PATH env var).
2. pip install -r requirements.txt
3. python app.py
4. Find this machine's local IP (e.g. `ipconfig getifaddr en0` on Mac,
   `ipconfig` on Windows, `hostname -I` on Linux) and put it into the
   ESP32 sketch as `serverUrl`, e.g. http://192.168.1.42:5000/predict

ENDPOINTS
---------
GET  /health   -> simple liveness check
POST /predict  -> multipart/form-data with field name "image" (jpeg)
                   returns JSON diagnosis
"""

import os
import io
import time
import logging
from datetime import datetime

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "cassava_model.keras")
IMG_SIZE = (224, 224)
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("farmbot")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model once at startup (kept warm in memory between requests)
# ---------------------------------------------------------------------------
log.info(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
log.info("Model loaded successfully.")


def classify_probability(prob: float):
    """
    Map the raw sigmoid output (0..1, higher = more 'sick') to a
    human-friendly status / severity / color, so the ESP32-CAM can
    color-code the TFT result card without doing any math itself.
    """
    if prob < 0.30:
        return {
            "status": "healthy",
            "label": "Healthy Cassava",
            "severity": "good",
            "color": "green",
            "advice": "No signs of disease. Keep up routine monitoring.",
        }
    elif prob < 0.50:
        return {
            "status": "healthy",
            "label": "Likely Healthy",
            "severity": "watch",
            "color": "yellow",
            "advice": "Mostly healthy but borderline. Re-check in a few days.",
        }
    elif prob < 0.75:
        return {
            "status": "sick",
            "label": "Early Signs of Disease",
            "severity": "mild",
            "color": "orange",
            "advice": "Possible early infection. Inspect leaves closely.",
        }
    else:
        return {
            "status": "sick",
            "label": "Sick Cassava",
            "severity": "severe",
            "color": "red",
            "advice": "Strong indication of disease. Consider treatment.",
        }


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image field in request. Use form field name 'image'."}), 400

    file = request.files["image"]
    file_bytes = file.read()

    if len(file_bytes) == 0:
        return jsonify({"error": "Empty image file."}), 400

    try:
        img_array = preprocess_image(file_bytes)
        prediction = model.predict(img_array, verbose=0)
        prob = float(prediction[0][0])
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    result = classify_probability(prob)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    confidence = (prob if result["status"] == "sick" else (1 - prob)) * 100

    response = {
        "label": result["label"],
        "status": result["status"],         # "healthy" or "sick"
        "severity": result["severity"],     # good / watch / mild / severe
        "color": result["color"],           # green / yellow / orange / red
        "probability": round(prob, 4),       # raw 0..1 model output
        "confidence": round(confidence, 1),  # 0..100, "how sure" of the verdict above
        "advice": result["advice"],
        "server_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inference_ms": elapsed_ms,
    }

    log.info(f"Prediction: {response['label']} (prob={response['probability']}, "
              f"conf={response['confidence']}%) in {elapsed_ms}ms")
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
