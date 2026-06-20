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
import sqlite3
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
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

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "farmbot.db"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("farmbot")

# static_url_path="" serves files in static/ directly at the root, e.g.
# static/index.html is reachable once we add an explicit "/" route below.
app = Flask(__name__, static_folder="static", static_url_path="")


# ---------------------------------------------------------------------------
# Database (scan history)
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT,
            status TEXT,
            severity TEXT,
            color TEXT,
            probability REAL,
            confidence REAL,
            advice TEXT,
            device_time TEXT,
            server_time TEXT,
            inference_ms REAL
        )
    """)
    conn.commit()
    conn.close()


init_db()

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

@app.route("/")
def serve_dashboard():
    return send_from_directory(app.static_folder, "index.html")

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

    # Optional field the ESP32-CAM sends alongside the image: its own
    # NTP-derived local capture time, so history matches what was shown
    # on the device's TFT at the moment of capture.
    device_time = request.form.get("captured_at", "").strip()

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
    server_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the image to disk and record this scan in history.
    filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    try:
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
            f.write(file_bytes)

        conn = get_db()
        conn.execute(
            """INSERT INTO scans
               (filename, label, status, severity, color, probability, confidence,
                advice, device_time, server_time, inference_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (filename, result["label"], result["status"], result["severity"], result["color"],
             round(prob, 4), round(confidence, 1), result["advice"], device_time,
             server_time, elapsed_ms),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Don't fail the ESP32's request just because logging/storage hiccuped.
        log.exception("Failed to persist scan history (prediction still returned)")

    response = {
        "label": result["label"],
        "status": result["status"],         # "healthy" or "sick"
        "severity": result["severity"],     # good / watch / mild / severe
        "color": result["color"],           # green / yellow / orange / red
        "probability": round(prob, 4),       # raw 0..1 model output
        "confidence": round(confidence, 1),  # 0..100, "how sure" of the verdict above
        "advice": result["advice"],
        "server_timestamp": server_time,
        "inference_ms": elapsed_ms,
    }

    log.info(f"Prediction: {response['label']} (prob={response['probability']}, "
              f"conf={response['confidence']}%) in {elapsed_ms}ms")
    return jsonify(response), 200


# ---------------------------------------------------------------------------
# History API (used by the dashboard frontend)
# ---------------------------------------------------------------------------
@app.route("/api/history", methods=["GET"])
def api_history():
    limit = min(int(request.args.get("limit", 24)), 200)
    offset = int(request.args.get("offset", 0))
    status_filter = request.args.get("status", "all")

    conn = get_db()
    if status_filter in ("healthy", "sick"):
        rows = conn.execute(
            "SELECT * FROM scans WHERE status = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (status_filter, limit, offset),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE status = ?", (status_filter,)
        ).fetchone()[0]
    else:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    conn.close()

    items = [{
        "id": r["id"],
        "image_url": f"/uploads/{r['filename']}",
        "label": r["label"],
        "status": r["status"],
        "severity": r["severity"],
        "color": r["color"],
        "probability": r["probability"],
        "confidence": r["confidence"],
        "advice": r["advice"],
        "device_time": r["device_time"],
        "server_time": r["server_time"],
        "inference_ms": r["inference_ms"],
    } for r in rows]

    return jsonify({"items": items, "total": total, "limit": limit, "offset": offset})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    healthy = conn.execute("SELECT COUNT(*) FROM scans WHERE status='healthy'").fetchone()[0]
    sick = conn.execute("SELECT COUNT(*) FROM scans WHERE status='sick'").fetchone()[0]
    last_row = conn.execute(
        "SELECT COALESCE(device_time, server_time) AS t FROM scans ORDER BY id DESC LIMIT 1"
    ).fetchone()
    avg_conf_row = conn.execute("SELECT AVG(confidence) FROM scans").fetchone()
    conn.close()

    return jsonify({
        "total": total,
        "healthy": healthy,
        "sick": sick,
        "healthy_pct": round((healthy / total * 100) if total else 0, 1),
        "sick_pct": round((sick / total * 100) if total else 0, 1),
        "avg_confidence": round(avg_conf_row[0], 1) if avg_conf_row[0] is not None else 0,
        "last_scan_time": last_row["t"] if last_row else None,
    })


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)





if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
