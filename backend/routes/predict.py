from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename

predict_bp = Blueprint("predict", __name__)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if file type is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Save file to uploads folder
def save_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path

@predict_bp.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type"}), 400

    try:
        file_path = save_file(file)
        return jsonify({
            "success": True,
            "message": "File uploaded successfully",
            "file_path": file_path
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500
