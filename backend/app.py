"""
backend/app.py
──────────────
Flask REST API for Brain Tumor Detection.

Endpoints
---------
GET  /health   → {"status": "ok", "model_loaded": true}
POST /predict  → multipart/form-data, field: images (one or many)
               ← JSON {results: [...], errors: [...]}
"""

from __future__ import annotations

import json
import os
import sys

# Resolve paths relative to project root (one level up from this file)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "brain_tumor_model.h5")
IDX_PATH   = os.path.join(BASE_DIR, "backend", "model", "class_indices.json")

# Add project root to sys.path so `from backend.utils...` works
sys.path.insert(0, BASE_DIR)

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.utils.preprocessing import (
    array_to_base64_png,
    load_image_from_bytes,
    make_gradcam_heatmap,
    overlay_gradcam,
    prepare_batch,
    preprocess_for_model,
)

# ─── Display name mapping ─────────────────────────────────────────────────────
DISPLAY_NAMES: dict[str, str] = {
    "glioma"    : "Glioma",
    "meningioma": "Meningioma",
    "notumor"   : "No Tumor",
    "pituitary" : "Pituitary Tumor",
}

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Model loading ────────────────────────────────────────────────────────────
model: tf.keras.Model | None = None
idx_to_label: dict[int, str] = {}
last_conv_layer: str = "top_conv"


def load_model_once() -> None:
    global model, idx_to_label, last_conv_layer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(IDX_PATH):
        raise FileNotFoundError(f"class_indices.json not found: {IDX_PATH}")

    print(f"[NeuraScan] Loading model from {MODEL_PATH} …")

    # ── FIX: use compile=False to avoid Keras version mismatch ───────────────
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                'BatchNormalization': tf.keras.layers.BatchNormalization
            }
        )
        print("[NeuraScan] Model loaded successfully.")
    except Exception as e1:
        print(f"[NeuraScan] First load attempt failed: {e1}")
        print("[NeuraScan] Trying tf_keras fallback...")
        try:
            import tf_keras
            model = tf_keras.models.load_model(MODEL_PATH, compile=False)
            print("[NeuraScan] Model loaded via tf_keras.")
        except Exception as e2:
            print(f"[NeuraScan] tf_keras also failed: {e2}")
            raise RuntimeError(
                f"Could not load model. Error 1: {e1} | Error 2: {e2}"
            )

    # Build reverse index: int → display name
    with open(IDX_PATH) as f:
        class_indices: dict[str, int] = json.load(f)
    idx_to_label = {v: DISPLAY_NAMES.get(k, k.capitalize())
                    for k, v in class_indices.items()}

    # Find last conv layer name automatically
    conv_layers = [l.name for l in model.layers
                   if isinstance(l, tf.keras.layers.Conv2D)]
    if conv_layers:
        last_conv_layer = conv_layers[-1]
    print(f"[NeuraScan] Grad-CAM layer: {last_conv_layer}")
    print(f"[NeuraScan] Classes: {idx_to_label}")
    print("[NeuraScan] Model ready.")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded. Use field name 'images'."}), 400

    results = []
    errors  = []

    for file in files:
        fname = file.filename or "unknown"
        try:
            raw         = file.read()
            img_arr     = load_image_from_bytes(raw)
            model_input = preprocess_for_model(img_arr)
            batch       = prepare_batch(model_input)

            # ── Inference ─────────────────────────────────────────────────
            preds    = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            conf     = float(preds[pred_idx])

            # ── Grad-CAM ──────────────────────────────────────────────────
            try:
                heatmap = make_gradcam_heatmap(
                    batch, model, last_conv_layer, pred_index=pred_idx
                )
                overlay     = overlay_gradcam(img_arr, heatmap)
                gradcam_b64 = array_to_base64_png(overlay)
            except Exception as gcam_err:
                print(f"[NeuraScan] Grad-CAM failed for {fname}: {gcam_err}")
                gradcam_b64 = array_to_base64_png(img_arr)

            # ── Probabilities ─────────────────────────────────────────────
            probabilities = {
                idx_to_label[i]: float(preds[i])
                for i in range(len(preds))
                if i in idx_to_label
            }

            results.append({
                "filename"       : fname,
                "predicted"      : idx_to_label.get(pred_idx, str(pred_idx)),
                "confidence"     : conf,
                "probabilities"  : probabilities,
                "original_image" : array_to_base64_png(img_arr),
                "gradcam_overlay": gradcam_b64,
            })

        except Exception as exc:
            errors.append({"filename": fname, "error": str(exc)})

    return jsonify({"results": results, "errors": errors})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model_once()
    app.run(host="0.0.0.0", port=5000, debug=False)