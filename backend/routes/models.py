from flask import Blueprint, jsonify, send_file
import os
import zipfile
import tempfile

models_bp = Blueprint("models", __name__)

# Absolute path — works regardless of where Flask is launched from
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@models_bp.route("/", methods=["GET"])
def list_models():
    if not os.path.exists(MODELS_DIR):
        return jsonify({"models": []})

    models = [
        folder for folder in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, folder))
    ]
    return jsonify({"models": models})


@models_bp.route("/download/<model_name>", methods=["GET"])
def download_model(model_name):
    """
    Zips the model folder (encoder.pth, autoencoder.pth,
    scaler.joblib, threshold.joblib) and returns it for download.
    """
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    # Build zip in a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(model_path):
            file_path = os.path.join(model_path, filename)
            if os.path.isfile(file_path):
                zf.write(file_path, arcname=filename)

    return send_file(
        tmp.name,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{model_name}.zip"
    )
