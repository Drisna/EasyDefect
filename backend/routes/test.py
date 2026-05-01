import sys
import os

# Ensure backend/ root is on the path so 'utils' can always be found
# regardless of how Flask is launched
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from utils.test_utils import test_images
from utils.user_context import get_request_user_email, get_user_storage_key

test_bp = Blueprint("test", __name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@test_bp.route("/", methods=["POST"])
def test_model():
    """
    Batch testing endpoint.

    Form fields:
        model_name       -- name of trained model folder inside models/
        normal_files     -- one or more normal images   (optional)
        defective_files  -- one or more defective images (optional)
    """
    model_name = request.form.get("model_name", "").strip()
    email = get_request_user_email()

    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    if not model_name:
        return jsonify({"error": "model_name is required"}), 400

    normal_files    = request.files.getlist("normal_files")
    defective_files = request.files.getlist("defective_files")

    # Filter out empty entries
    normal_files    = [f for f in normal_files    if f and f.filename]
    defective_files = [f for f in defective_files if f and f.filename]

    if not normal_files and not defective_files:
        return jsonify({"error": "Upload at least one image to test"}), 400

    try:
        user_models_dir = os.path.join(MODELS_DIR, get_user_storage_key(email))
        results = test_images(model_name, normal_files, defective_files, user_model_root=user_models_dir)
        return jsonify(results)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    except Exception as e:
        print("Testing Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
