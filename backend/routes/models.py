from flask import Blueprint, jsonify
import os

models_bp = Blueprint("models", __name__)

@models_bp.route("/", methods=["GET"])
def list_models():
    models_dir = "models"

    if not os.path.exists(models_dir):
        return jsonify({"models": []})

    models = [
        folder for folder in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, folder))
    ]

    return jsonify({"models": models})
