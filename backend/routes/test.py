# from flask import Blueprint, request, jsonify
# from utils.test_utils import test_images

# test_bp = Blueprint("test", __name__)

# @test_bp.route("/", methods=["POST"])
# def test_model():
#     model_name = request.form.get("model_name")

#     if not model_name:
#         return jsonify({"error": "Model name required"}), 400

#     normal_files = request.files.getlist("normal_files")
#     defective_files = request.files.getlist("defective_files")

#     if not normal_files and not defective_files:
#         return jsonify({"error": "No files uploaded"}), 400

#     results = test_images(model_name, normal_files, defective_files)

#     return jsonify(results)

from flask import Blueprint, request, jsonify
from utils.test_utils import test_images

test_bp = Blueprint("test", __name__)


@test_bp.route("/", methods=["POST"])
def test_model():
    model_name = request.form.get("model_name")

    if not model_name:
        return jsonify({"error": "Model name required"}), 400

    normal_files = request.files.getlist("normal_files")
    defective_files = request.files.getlist("defective_files")

    if not normal_files and not defective_files:
        return jsonify({"error": "No files uploaded"}), 400

    try:
        results = test_images(model_name, normal_files, defective_files)
        return jsonify(results)

    except Exception as e:
        print("❌ Testing Error:", str(e))
        return jsonify({"error": str(e)}), 500
