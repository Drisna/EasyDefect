import os
import sys
import traceback
import uuid
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from utils.user_context import get_request_user_email, get_user_storage_key

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

predict_bp = Blueprint("predict", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Track whether we've cleared the folder for this training session.
# Set to False whenever /clear is called, True after first upload.
_session_started = False

print(f"[predict] Upload folder: {UPLOAD_FOLDER}")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _user_upload_folder(email):
    user_key = get_user_storage_key(email)
    folder = os.path.join(UPLOAD_FOLDER, user_key)
    os.makedirs(folder, exist_ok=True)
    return folder


@predict_bp.route("/", methods=["POST"])
def upload_image():
    """
    Receives a single training image and saves it to the uploads folder.
    Called by TrainingPage.js for each image selected.
    """
    global _session_started
    email = get_request_user_email()
    if not email:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"Invalid file type. Allowed: jpg, jpeg, png, bmp"
        }), 400

    try:
        # ── KEY FIX: seek to start before saving ─────────────────────────
        # Ensures the full file is written even if stream was partially read
        file.stream.seek(0)

        original = secure_filename(file.filename)
        stem, ext = os.path.splitext(original)
        filename = f"{stem}_{uuid.uuid4().hex[:8]}{ext.lower()}"
        file_path = os.path.join(_user_upload_folder(email), filename)
        file.save(file_path)

        print(f"[predict] Saved: {filename}  ({os.path.getsize(file_path)} bytes)")

        return jsonify({
            "success":   True,
            "message":   "File uploaded successfully",
            "file_path": file_path
        })

    except Exception as e:
        print(f"[predict] EXCEPTION:")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500


@predict_bp.route("/clear", methods=["DELETE"])
def clear_uploads():
    """
    Delete all files in the uploads folder.
    Called from TrainingPage after training completes,
    and from TrainingPage on component mount to clear stale images.
    """
    global _session_started
    email = get_request_user_email()
    if not email:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    _session_started = False

    try:
        deleted = 0
        user_folder = _user_upload_folder(email)
        for filename in os.listdir(user_folder):
            file_path = os.path.join(user_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted += 1

        print(f"[predict] Cleared {deleted} files from uploads folder")
        return jsonify({"success": True, "message": f"Cleared {deleted} files"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500