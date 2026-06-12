"""
backend/utils/preprocessing.py
────────────────────────────────
Image loading, resizing, normalisation and Grad-CAM heatmap generation.
"""

from __future__ import annotations

import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as _efficientnet_preprocess_input


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224          # must match training


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_image_from_bytes(raw: bytes) -> np.ndarray:
    """
    Accept raw file bytes → RGB PIL image → float32 numpy array (H, W, 3).
    Pixel values are in [0, 1] — for display/visualization only.
    Pass through preprocess_for_model() before feeding to the model.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_for_model(arr: np.ndarray) -> np.ndarray:
    """
    Apply EfficientNetB0 preprocessing to a [0, 1] float32 array.

    Matches training: ImageDataGenerator(preprocessing_function=preprocess_input)
    which expects [0, 255] and applies ImageNet mean/std normalisation
    (torch mode: x/255, then subtract mean, divide by std).
    """
    arr_255 = arr * 255.0
    return _efficientnet_preprocess_input(arr_255)


def prepare_batch(arr: np.ndarray) -> np.ndarray:
    """Add batch dimension: (H, W, C) → (1, H, W, C)."""
    return np.expand_dims(arr, axis=0)


def array_to_base64_png(arr: np.ndarray) -> str:
    """
    Convert a float [0,1] or uint8 numpy array to a base64-encoded PNG string
    suitable for embedding in JSON (data:image/png;base64,...).
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Parameters
    ----------
    img_array           : shape (1, H, W, C), float32 in [0, 1]
    model               : full Keras model
    last_conv_layer_name: name of the last convolutional layer in the backbone
    pred_index          : class index to explain (defaults to argmax)

    Returns
    -------
    heatmap : float32 array (H, W) in [0, 1]
    """
    # Sub-model that outputs (conv_output, final_predictions)
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output,
                   model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the class score w.r.t. conv feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions → importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    original_arr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_arr : (H, W, 3) float32 in [0, 1]
    heatmap      : (h, w)  float32 in [0, 1]   (may differ in size)
    alpha        : heatmap opacity
    colormap     : matplotlib colormap name

    Returns
    -------
    overlay : (H, W, 3) uint8
    """
    import matplotlib.cm as cm

    # Resize heatmap to match original
    H, W = original_arr.shape[:2]
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_img = heatmap_img.resize((W, H), Image.LANCZOS)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_resized)[:, :, :3]   # drop alpha channel

    # Blend
    original_rgb = np.clip(original_arr, 0, 1)
    overlay = (1 - alpha) * original_rgb + alpha * colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay
