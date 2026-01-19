from flask import Blueprint, request, jsonify
import os
from utils.train_utils import train_anomaly_detector

train_bp = Blueprint("train", __name__)

@train_bp.route("/", methods=["POST"])
def train_model():
    """
    Expects JSON:
    {
        "model_name": "my_model",
        "epochs": 50
    }
    """
    data = request.get_json()
    model_name = data.get("model_name")
    epochs = int(data.get("epochs", 50))

    if not model_name:
        return jsonify({"success": False, "message": "Model name is required"}), 400

    dataset_path = "uploads"          # images uploaded by manufacturer
    model_save_path = os.path.join("models", model_name)

    try:
        train_anomaly_detector(dataset_path, model_save_path, epochs)
        return jsonify({"success": True, "message": f"Model trained and saved to {model_save_path}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
