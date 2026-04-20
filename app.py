"""
app.py — PulmoScan Flask server (Railway production version)

Changes from local version:
  - Supports both .h5 and .keras model formats
  - Uses PORT environment variable (Railway requirement)
  - Secret key from environment variable
  - Temp folders for uploads/results (Railway has ephemeral filesystem)
  - Gunicorn compatible (no debug mode)
"""

import os
import uuid
import json
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from werkzeug.utils import secure_filename

# ── ML imports ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from PIL import Image
    import cv2
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Support both .h5 and .keras formats
MODEL_PATH = None
for candidate in [
    BASE_DIR / "model" / "best_model.h5",
    BASE_DIR / "model" / "best_model.keras",
]:
    if candidate.exists():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    MODEL_PATH = BASE_DIR / "model" / "best_model.h5"  # default path

# Use temp dir on Railway (ephemeral filesystem)
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "pulmoscan_uploads"
RESULT_FOLDER = Path(tempfile.gettempdir()) / "pulmoscan_results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}
IMG_SIZE    = (224, 224)
IMG_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD   = 0.5

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── model singleton ───────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    if not ML_AVAILABLE:
        return None
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}")
        return None
    logger.info(f"Loading model from {MODEL_PATH}...")
    _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    _model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("Model loaded successfully.")
    return _model

# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return np.expand_dims(arr, axis=0)


def generate_gradcam(model, image_path, save_path):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv = sub
                        break
                if last_conv:
                    break

        if last_conv is None:
            return ""

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

        inp = tf.cast(preprocess(image_path), tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp)
            loss = preds[:, 0]

        grads  = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
        cam    = np.maximum(cam, 0)
        cam   /= (cam.max() + 1e-8)

        cam_up  = cv2.resize(cam, IMG_SIZE)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig    = np.array(Image.open(image_path).convert("RGB").resize(IMG_SIZE))
        overlay = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)

        Image.fromarray(overlay).save(save_path)
        return save_path
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        return ""

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = MODEL_PATH.exists() and ML_AVAILABLE
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, BMP, or TIFF."}), 400

    uid      = uuid.uuid4().hex[:10]
    ext      = f.filename.rsplit(".", 1)[1].lower()
    filename = f"{uid}.{ext}"
    upload_p = UPLOAD_FOLDER / filename
    f.save(str(upload_p))

    model = get_model()
    if model is None:
        mock_prob = float(np.random.uniform(0.2, 0.8))
        label     = 1 if mock_prob >= THRESHOLD else 0
        conf      = mock_prob if label == 1 else 1 - mock_prob
        result = {
            "filename":    f.filename,
            "uid":         uid,
            "probability": round(mock_prob, 4),
            "label":       label,
            "confidence":  round(conf * 100, 1),
            "finding":     "Disease / TB Detected" if label == 1 else "Normal",
            "gradcam_url": None,
            "upload_url":  f"/uploads/{filename}",
            "time_ms":     round((time.time() - t0) * 1000),
            "demo_mode":   True,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        inp   = preprocess(str(upload_p))
        prob  = float(model.predict(inp, verbose=0)[0][0])
        label = 1 if prob >= THRESHOLD else 0
        conf  = prob if label == 1 else 1 - prob

        cam_filename = f"{uid}_cam.png"
        cam_path     = str(RESULT_FOLDER / cam_filename)
        cam_url      = None
        if generate_gradcam(model, str(upload_p), cam_path):
            cam_url = f"/results/{cam_filename}"

        result = {
            "filename":    f.filename,
            "uid":         uid,
            "probability": round(prob, 4),
            "label":       label,
            "confidence":  round(conf * 100, 1),
            "finding":     "Disease / TB Detected" if label == 1 else "Normal",
            "gradcam_url": cam_url,
            "upload_url":  f"/uploads/{filename}",
            "time_ms":     round((time.time() - t0) * 1000),
            "demo_mode":   False,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    history = session.get("history", [])
    history.insert(0, result)
    session["history"] = history[:20]
    return jsonify(result)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(str(RESULT_FOLDER), filename)


@app.route("/history")
def history():
    return jsonify(session.get("history", []))


@app.route("/model-status")
def model_status():
    return jsonify({
        "model_found":  MODEL_PATH.exists(),
        "ml_available": ML_AVAILABLE,
        "ready":        MODEL_PATH.exists() and ML_AVAILABLE,
        "training_info": {
            "datasets":     ["Montgomery", "Shenzhen", "Darwin"],
            "total_images": 6810,
            "montgomery":   138,
            "shenzhen":     566,
            "darwin":       6106,
            "val_accuracy": 0.93,
            "val_auc":      0.982,
            "model":        "VGG16 Transfer Learning",
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
