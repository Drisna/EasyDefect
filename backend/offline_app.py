import os
import sys
import threading
import time
import webbrowser

import joblib
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

try:
    from openvino.runtime import Core
except Exception:
    from openvino import Core

from utils.offline_bundle_html import OFFLINE_HOME_HTML, OFFLINE_TEST_HTML


def app_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_base_dir()
MODEL_DIR = os.path.join(BASE_DIR, "model")
DISPLAY_NAME_FILE = os.path.join(BASE_DIR, "display_name.txt")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)

core = None
encoder = None
autoencoder = None
threshold = None
enc_output = None
ae_output = None


def read_display_name():
    try:
        with open(DISPLAY_NAME_FILE, "r", encoding="utf-8") as f:
            name = f.read().strip()
            return name or "User"
    except OSError:
        return "User"


def require_model_file(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {filename}")
    return path


def load_model():
    global core, encoder, autoencoder, threshold, enc_output, ae_output
    if encoder is not None and autoencoder is not None:
        return

    core = Core()
    encoder = core.compile_model(require_model_file("encoder.xml"), "CPU")
    autoencoder = core.compile_model(require_model_file("autoencoder.xml"), "CPU")
    threshold = float(joblib.load(require_model_file("threshold.joblib")))
    enc_output = encoder.output(0)
    ae_output = autoencoder.output(0)


def preprocess_image(img):
    img = img.convert("RGB").resize((256, 256), Image.BILINEAR)
    offset = (256 - 224) // 2
    img = img.crop((offset, offset, offset + 224, offset + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict_image(img):
    load_model()
    arr = preprocess_image(img)
    feat = encoder([arr])[enc_output]
    feat = np.squeeze(feat, axis=0)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    feat = feat.astype(np.float32).reshape(1, -1)
    recon = autoencoder([feat])[ae_output]
    error = float(np.mean((recon - feat) ** 2))
    label = "Normal" if error <= threshold else "Defective"
    return label, error


@app.get("/")
def home():
    return OFFLINE_HOME_HTML.replace("__DISPLAY_NAME__", read_display_name())


@app.get("/index.html")
def index_html():
    return home()


@app.get("/test.html")
def test_html():
    return OFFLINE_TEST_HTML


@app.get("/health")
def health():
    try:
        load_model()
        return jsonify({"ok": True, "message": "EasyDefect offline runner is ready."})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.post("/predict")
def predict():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "Upload at least one image to test"}), 400

    results = []
    for file in files:
        try:
            img = Image.open(file.stream)
            prediction, error = predict_image(img)
            results.append({
                "filename": file.filename,
                "prediction": prediction,
                "error": round(error, 8),
                "threshold": round(float(threshold), 8),
            })
        except Exception as exc:
            results.append({
                "filename": file.filename,
                "prediction": "Error",
                "error": None,
                "threshold": round(float(threshold), 8) if threshold is not None else None,
                "message": str(exc),
            })

    return jsonify({"total": len(results), "results": results})


def open_browser_later(port):
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{port}/")


def main():
    port = int(os.environ.get("EASYDEFECT_OFFLINE_PORT", "8000"))
    threading.Thread(target=open_browser_later, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
