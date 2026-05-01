from flask import Blueprint, request, jsonify
import os
from utils.train_utils import train_anomaly_detector
from utils.user_context import get_request_user_email, get_user_storage_key

train_bp = Blueprint("train", __name__)

# Absolute paths — works regardless of where Flask is launched from
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
MODELS_DIR  = os.path.join(BASE_DIR, "models")


@train_bp.route("/", methods=["POST"])
def train_model():
    """
    Expects JSON:
    {
        "model_name": "my_model",
        "epochs": 50
    }
    Images must have been uploaded to /api/predict/ first.
    """
    data       = request.get_json() or {}
    model_name = data.get("model_name", "").strip()
    epochs     = int(data.get("epochs", 50))
    email = get_request_user_email()

    if not email:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    if not model_name:
        return jsonify({"success": False, "message": "model_name is required"}), 400

    # Count valid images in uploads folder
    user_upload_dir = os.path.join(UPLOAD_DIR, get_user_storage_key(email))
    if not os.path.exists(user_upload_dir):
        return jsonify({"success": False, "message": "No images uploaded yet"}), 400

    valid_exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    image_count = sum(
        1 for f in os.listdir(user_upload_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    )

    if image_count < 20:
        return jsonify({
            "success": False,
            "message": f"At least 20 images required. Found: {image_count}"
        }), 400

    user_models_dir = os.path.join(MODELS_DIR, get_user_storage_key(email))
    model_save_path = os.path.join(user_models_dir, model_name)

    try:
        train_anomaly_detector(user_upload_dir, model_save_path, epochs)
        return jsonify({
            "success": True,
            "message": f"Model '{model_name}' trained successfully with {image_count} images."
        })
    except Exception as e:
        print("❌ Training Error:", str(e))
        return jsonify({"success": False, "message": str(e)}), 500
