"""
Interface Web Flask - Détection Feuille d'Olivier
Usage:
    python app.py
    Puis ouvrir: http://localhost:5000
"""

import os
import io
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# ── Modèles ────────────────────────────────────────────────
MODEL_PATHS = {1: "peacock_model1.h5", 2: "peacock_model2.h5"}
MODEL_NAMES = {1: "CNN Basique (~85%)", 2: "MobileNetV2 (~95%)"}
IMG_SIZE = 224

loaded_models = {}

def get_model(num):
    if num not in loaded_models:
        path = MODEL_PATHS[num]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle {path} introuvable.")
        loaded_models[num] = load_model(path)
    return loaded_models[num]

# Précharger le modèle 2 au démarrage
try:
    get_model(2)
    print("✅ Modèle 2 (MobileNetV2) chargé.")
except Exception as e:
    print(f"⚠️  Erreur chargement modèle 2: {e}")

# ── Historique ─────────────────────────────────────────────
history = []

# ── Prédiction ─────────────────────────────────────────────
def predict(image_bytes, model_num=2, thr_high=0.75, thr_low=0.25):
    arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Image illisible.")

    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input   = np.expand_dims(img_resized, axis=0)

    mdl  = get_model(model_num)
    conf = float(mdl.predict(img_input, verbose=0)[0][0])

    if conf > thr_high:
        label  = "Malade : Œil de Paon"
        status = "INFECTED"
        color  = "#e74c3c"
        emoji  = "🔴"
    elif conf < thr_low:
        label  = "Feuille Saine"
        status = "HEALTHY"
        color  = "#27ae60"
        emoji  = "🟢"
    else:
        label  = "Incertain"
        status = "UNCERTAIN"
        color  = "#f39c12"
        emoji  = "🟡"

    # Encode image pour affichage dans le navigateur
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "label":      label,
        "status":     status,
        "confidence": round(conf, 4),
        "confidence_pct": f"{conf*100:.1f}%",
        "color":      color,
        "emoji":      emoji,
        "model_num":  model_num,
        "model_name": MODEL_NAMES[model_num],
        "image_b64":  img_b64,
        "timestamp":  datetime.now().strftime("%H:%M:%S"),
        "id":         str(uuid.uuid4())[:8],
    }

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Fichier vide."}), 400

    model_num  = int(request.form.get("model", 2))
    thr_high   = float(request.form.get("threshold_high", 0.75))
    thr_low    = float(request.form.get("threshold_low",  0.25))

    try:
        result = predict(f.read(), model_num, thr_high, thr_low)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Erreur: {e}"}), 500

    # Ajouter à l'historique (max 10)
    entry = {k: v for k, v in result.items() if k != "image_b64"}
    entry["filename"] = f.filename
    history.insert(0, entry)
    if len(history) > 10:
        history.pop()

    return jsonify(result)

@app.route("/history")
def api_history():
    return jsonify(history)

@app.route("/models")
def api_models():
    available = {}
    for num, path in MODEL_PATHS.items():
        available[num] = {
            "name":      MODEL_NAMES[num],
            "available": os.path.exists(path),
            "loaded":    num in loaded_models,
        }
    return jsonify(available)

if __name__ == "__main__":
    print("🌿 Olive Leaf Detector - Interface Web")
    print("📡 Démarrage sur http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
