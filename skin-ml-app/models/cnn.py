"""
CNN page — Image-based skin-type classifier.

This module loads the pre-trained TensorFlow CNN model from disk and runs
inference.  No training happens at runtime — the model was trained once by
running  skin-ml-app/train_cnn.py  and saved to:

    skin-ml-app/models/saved_cnn/skin_type_cnn.h5

If the saved model does not exist yet, this module raises a clear error
telling you to run the training script first.
"""

import os
import io
import json
import base64
import logging
import traceback
import numpy as np

# Must be set before any tensorflow import — forces Keras 2 legacy backend on TF 2.15+
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "saved_cnn")
MODEL_PATH = os.path.join(SAVE_DIR, "skin_type_cnn.h5")
META_PATH  = os.path.join(SAVE_DIR, "meta.json")

# ── Load metadata (classes, img_size, …) ─────────────────────────────────────
def _load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)

_META = _load_meta()
CLASSES  = _META["classes"]  if _META else ["combination", "dry", "normal", "oily"]
IMG_SIZE = _META["img_size"] if _META else 128

CLASS_COLORS = {
    "combination": "#6c63ff",
    "dry":         "#38bdf8",
    "normal":      "#00d4aa",
    "oily":        "#ff6b6b",
}

# ── Lazy-load the TF model (loaded once, cached in module scope) ───────────────
_TF_MODEL = None

def _get_model():
    global _TF_MODEL
    if _TF_MODEL is not None:
        return _TF_MODEL

    logger.info("[CNN] Model path (absolute): %s", MODEL_PATH)
    logger.info("[CNN] Model file exists: %s", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained CNN model not found at: {MODEL_PATH}\n"
            f"Run: python skin-ml-app/train_cnn.py"
        )

    logger.info("[CNN] Loading TensorFlow model ...")
    print("[CNN] Loading TensorFlow model ...", flush=True)
    try:
        import tensorflow as tf
        logger.info("[CNN] TensorFlow version: %s", tf.__version__)
        _TF_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        val_acc = _META.get("val_acc", "?") if _META else "?"
        msg = f"[CNN] Model loaded OK (val_acc={float(val_acc)*100:.1f}%)"
        logger.info(msg)
        print(msg, flush=True)
    except Exception:
        logger.error("[CNN] FAILED to load model:\n%s", traceback.format_exc())
        raise
    return _TF_MODEL


# ── Image pre-processing (single upload) ─────────────────────────────────────
def preprocess_image(file_bytes: bytes):
    """
    Returns:
        arr_norm : np.ndarray  shape (1, IMG_SIZE, IMG_SIZE, 3)  float32 in [0,1]
        thumb_b64: str         base64-encoded JPEG thumbnail for display
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = arr[np.newaxis, ...]  # add batch dim → (1, H, W, 3)

    # Thumbnail for display
    thumb = img.resize((224, 224))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()
    return arr, thumb_b64

# ── Visualisations ────────────────────────────────────────────────────────────
BG = "#050510"

def _b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _plot_donut_chart(proba_dict):
    """Donut chart of class probabilities."""
    labels = list(proba_dict.keys())
    sizes = [proba_dict[l] for l in labels]
    
    # Clean up very tiny probabilities
    filtered_labels = [l.title() for l, s in zip(labels, sizes) if s > 0.5]
    filtered_sizes = [s for s in sizes if s > 0.5]
    
    colors = [CLASS_COLORS.get(l.lower(), "#ffffff") for l in filtered_labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(BG)
    
    wedges, texts, autotexts = ax.pie(
        filtered_sizes, labels=filtered_labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.80, 
        wedgeprops=dict(width=0.4, edgecolor=BG, linewidth=2),
        textprops=dict(color="#a0a8c8", fontsize=10, fontweight="600")
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax.set_title("Probability Distribution", color="white", fontsize=12, fontweight="700", pad=15)
    fig.tight_layout()
    return _b64_fig(fig)


def _plot_proba_radar(proba_dict):
    """Radar / spider chart of class probabilities."""
    labels = list(proba_dict.keys())
    vals   = [proba_dict[l] / 100 for l in labels]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   += vals[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#0d0d1f")
    ax.spines["polar"].set_color("#2a2a55")
    ax.tick_params(colors="#6272a4", labelsize=9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.title() for l in labels], color="#a0a8c8", fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(color="#1e1e40", linewidth=0.6)

    ax.plot(angles, vals, "o-", linewidth=2, color="#6c63ff")
    ax.fill(angles, vals, alpha=0.25, color="#6c63ff")

    max_i = vals.index(max(vals[:-1]))
    ax.plot(angles[max_i], vals[max_i], "o", markersize=12, color="#ffd93d", zorder=5)

    ax.set_title("Class Probability Radar", color="white",
                 fontsize=11, fontweight="700", pad=18)
    fig.tight_layout()
    return _b64_fig(fig)


# ── Main predict function (called by Flask) ────────────────────────────────────
def predict_cnn(file_bytes: bytes, params=None):
    """
    Loads the saved model (once) and returns a prediction dict.
    params is accepted for API compatibility but ignored —
    the architecture is fixed in the saved model.
    """
    model = _get_model()

    img_arr, thumb_b64 = preprocess_image(file_bytes)

    # Run inference
    proba_raw = model.predict(img_arr, verbose=0)[0]       # shape (4,)
    pred_idx  = int(np.argmax(proba_raw))
    predicted  = CLASSES[pred_idx]
    confidence = round(float(proba_raw[pred_idx]) * 100, 1)

    proba_dict = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(CLASSES, proba_raw)
    }

    # Info about the saved model
    val_acc    = round(_META.get("val_acc", 0) * 100, 1) if _META else "N/A"
    n_samples  = _META.get("trained_on_samples", "N/A") if _META else "N/A"

    return {
        "skin_type":    predicted,
        "confidence":   confidence,
        "probabilities": proba_dict,
        "train_acc":    val_acc,          # val accuracy from training run
        "n_iter":       n_samples,        # reused field — now shows training sample count
        "image_b64":    thumb_b64,
        "plot_donut":       _plot_donut_chart(proba_dict),
        "plot_radar":       _plot_proba_radar(proba_dict),
    }
