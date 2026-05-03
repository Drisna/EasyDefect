from html import escape
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from flask import Blueprint, after_this_request, jsonify, request, send_file
import joblib
import torch
import torch.nn as nn

from routes.auth import get_registered_user_display_name
from utils.user_context import get_request_user_email, get_user_storage_key


models_bp = Blueprint("models", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OFFLINE_DIST_DIR = os.path.join(BASE_DIR, "offline_dist", "EasyDefect_Offline")
OFFLINE_EXE_NAME = "EasyDefect_Offline.exe" if os.name == "nt" else "EasyDefect_Offline"

README_TXT = """EasyDefect Offline Bundle
========================

This package includes:
1. Your trained model converted to OpenVINO
2. A no-code offline testing interface
3. A PyInstaller desktop runner with Python/OpenVINO bundled inside

How to use:
-----------
1. Extract this zip to any folder.
2. Double-click Open_Testing_Page.bat.
3. Your browser opens automatically.
4. Pick product images and click Run test.

No Python install is required on the testing computer.
The app runs locally at http://127.0.0.1:8000 and does not need internet.

Do not delete the model folder or _internal folder; the executable needs them.
"""

OPEN_TESTING_PAGE_BAT = """@echo off
cd /d "%~dp0"
start "" "%~dp0EasyDefect_Offline.exe"
"""

OPEN_TESTING_PAGE_VBS = '''Set oShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
oShell.CurrentDirectory = scriptDir
oShell.Run """" & scriptDir & "\\EasyDefect_Offline.exe" & """", 0, False
'''


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _safe_model_name(model_name):
    model_name = (model_name or "").strip()
    if not model_name or model_name != os.path.basename(model_name):
        raise ValueError("Invalid model name")
    return model_name


def _ensure_offline_runner():
    exe_path = os.path.join(OFFLINE_DIST_DIR, OFFLINE_EXE_NAME)
    source_paths = [
        os.path.join(BASE_DIR, "offline_app.py"),
        os.path.join(BASE_DIR, "offline_app.spec"),
        os.path.join(BASE_DIR, "utils", "offline_bundle_html.py"),
    ]
    if os.path.exists(exe_path) and all(
        os.path.getmtime(exe_path) >= os.path.getmtime(path)
        for path in source_paths
        if os.path.exists(path)
    ):
        return OFFLINE_DIST_DIR

    spec_path = os.path.join(BASE_DIR, "offline_app.spec")
    if not os.path.exists(spec_path):
        raise RuntimeError("offline_app.spec not found. Cannot build offline runner.")

    try:
        import PyInstaller  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyInstaller is not installed on the backend. "
            "Install backend requirements, then retry download: pip install -r requirements.txt"
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        os.path.join(BASE_DIR, "offline_dist"),
        "--workpath",
        os.path.join(BASE_DIR, "offline_build"),
        spec_path,
    ]
    proc = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"PyInstaller failed to build offline runner. {details[-3000:]}")

    if not os.path.exists(exe_path):
        raise RuntimeError("PyInstaller finished but EasyDefect_Offline executable was not created.")

    return OFFLINE_DIST_DIR


def _export_openvino_model(model_path, export_dir):
    try:
        from torchvision.models import resnet50
    except Exception as exc:
        raise RuntimeError(
            "torchvision is missing in backend environment. Install with: pip install torchvision"
        ) from exc

    try:
        from openvino import convert_model, save_model
    except Exception:
        try:
            from openvino.tools.ovc import convert_model
            from openvino.runtime import serialize as save_model
        except Exception as exc:
            raise RuntimeError(
                "OpenVINO is missing in backend environment. Install with: pip install openvino"
            ) from exc

    device = torch.device("cpu")

    feature_model = resnet50(weights=None)
    feature_model.fc = nn.Identity()
    feature_model.load_state_dict(
        torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    )
    feature_model.eval()

    autoencoder = Autoencoder(input_dim=2048)
    autoencoder.load_state_dict(
        torch.load(os.path.join(model_path, "autoencoder.pth"), map_location=device)
    )
    autoencoder.eval()

    encoder_ov = convert_model(feature_model, example_input=torch.randn(1, 3, 224, 224))
    autoencoder_ov = convert_model(autoencoder, example_input=torch.randn(1, 2048))

    os.makedirs(export_dir, exist_ok=True)
    save_model(encoder_ov, os.path.join(export_dir, "encoder.xml"))
    save_model(autoencoder_ov, os.path.join(export_dir, "autoencoder.xml"))
    shutil.copy2(
        os.path.join(model_path, "threshold.joblib"),
        os.path.join(export_dir, "threshold.joblib"),
    )


def _copy_runner_to_bundle(runner_dir, bundle_dir):
    for item in os.listdir(runner_dir):
        src = os.path.join(runner_dir, item)
        dst = os.path.join(bundle_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _display_name_for_download(email):
    display_name = (request.args.get("display_name") or "").strip()
    if not display_name:
        display_name = get_registered_user_display_name(email)
    if not display_name and email and "@" in email:
        display_name = email.split("@", 1)[0].strip()
    return display_name or "User"


@models_bp.route("/", methods=["GET"])
def list_models():
    email = get_request_user_email()
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    user_models_dir = os.path.join(MODELS_DIR, get_user_storage_key(email))
    if not os.path.exists(user_models_dir):
        return jsonify({"models": []})

    models = [
        folder for folder in os.listdir(user_models_dir)
        if os.path.isdir(os.path.join(user_models_dir, folder))
    ]
    return jsonify({"models": models})


@models_bp.route("/download/<model_name>", methods=["GET"])
def download_model(model_name):
    """
    Builds a customer-ready zip containing:
    - OpenVINO model artifacts for the selected model
    - PyInstaller offline runner
    - no-code launcher scripts
    """
    try:
        model_name = _safe_model_name(model_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    email = get_request_user_email() or (request.args.get("user_email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    user_models_dir = os.path.join(MODELS_DIR, get_user_storage_key(email))
    model_path = os.path.join(user_models_dir, model_name)

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    required = ["encoder.pth", "autoencoder.pth", "threshold.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        return jsonify({"error": f"Model '{model_name}' is missing files: {missing}"}), 400

    tmp_root = tempfile.mkdtemp(prefix=f"easydefect_{model_name}_")
    bundle_dir = os.path.join(tmp_root, f"{model_name}_offline")
    model_export_dir = os.path.join(bundle_dir, "model")
    os.makedirs(bundle_dir, exist_ok=True)

    try:
        runner_dir = _ensure_offline_runner()
        _copy_runner_to_bundle(runner_dir, bundle_dir)
        _export_openvino_model(model_path, model_export_dir)

        with open(os.path.join(bundle_dir, "display_name.txt"), "w", encoding="utf-8") as f:
            f.write(escape(_display_name_for_download(email)))
        with open(os.path.join(bundle_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(README_TXT)
        with open(os.path.join(bundle_dir, "Open_Testing_Page.bat"), "w", encoding="utf-8") as f:
            f.write(OPEN_TESTING_PAGE_BAT)
        with open(os.path.join(bundle_dir, "Open_Testing_Page.vbs"), "w", encoding="utf-8") as f:
            f.write(OPEN_TESTING_PAGE_VBS)

        fd, zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(bundle_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, bundle_dir)
                    zf.write(file_path, arcname=arcname)
    except RuntimeError as err:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return jsonify({"error": str(err)}), 500
    except Exception as err:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return jsonify({"error": f"Failed to prepare model bundle: {err}"}), 500
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    @after_this_request
    def cleanup_zip(response):
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return response

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{model_name}_easydefect_offline.zip",
    )
