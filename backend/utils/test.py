from flask import Blueprint, request, jsonify
from utils.test_utils import test_images

test_bp = Blueprint("test", __name__)


@test_bp.route("/", methods=["POST"])
def test_model():
    """
    Batch testing endpoint.

    Form fields:
        model_name       — name of trained model folder inside models/
        normal_files     — one or more normal images   (optional)
        defective_files  — one or more defective images (optional)
    """
    model_name = request.form.get("model_name", "").strip()

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
        results = test_images(model_name, normal_files, defective_files)
        return jsonify(results)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    except Exception as e:
        print("❌ Testing Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
