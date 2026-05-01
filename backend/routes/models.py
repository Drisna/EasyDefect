from html import escape
from flask import Blueprint, jsonify, request, send_file
import os
import zipfile
import tempfile
import shutil
import torch
import torch.nn as nn
import joblib
from routes.auth import get_registered_user_display_name
from utils.offline_bundle_html import OFFLINE_HOME_HTML, OFFLINE_TEST_HTML
from utils.user_context import get_request_user_email, get_user_storage_key

models_bp = Blueprint("models", __name__)

# Absolute path — works regardless of where Flask is launched from
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

RUN_SERVER_PY = """import os
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from openvino.runtime import Core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)

core = Core()
encoder = core.compile_model(os.path.join(MODEL_DIR, "encoder.xml"), "CPU")
autoencoder = core.compile_model(os.path.join(MODEL_DIR, "autoencoder.xml"), "CPU")
threshold = float(joblib.load(os.path.join(MODEL_DIR, "threshold.joblib")))

enc_input = encoder.input(0)
enc_output = encoder.output(0)
ae_input = autoencoder.input(0)
ae_output = autoencoder.output(0)

def preprocess_image(img):
    # Match training preprocess: Resize(256) -> CenterCrop(224) -> Normalize
    img = img.resize((256, 256), Image.BILINEAR)
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)    # NCHW
    return arr.astype(np.float32)

def predict(img):
    arr = preprocess_image(img)
    feat = encoder([arr])[enc_output]
    feat = np.squeeze(feat, axis=0)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    feat = feat.astype(np.float32).reshape(1, -1)
    recon = autoencoder([feat])[ae_output]
    error = float(np.mean((recon - feat) ** 2))
    label = "Normal" if error <= threshold else "Defective"
    return label, error

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/test.html")
def test_page():
    return send_from_directory(BASE_DIR, "test.html")

@app.post("/predict")
def run_predict():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "Upload at least one image to test"}), 400
    try:
        results = []
        for f in files:
            try:
                img = Image.open(f.stream).convert("RGB")
                pred, err = predict(img)
                results.append({
                    "filename": f.filename,
                    "prediction": pred,
                    "error": round(err, 8),
                    "threshold": round(threshold, 8),
                })
            except Exception:
                results.append({
                    "filename": f.filename,
                    "prediction": "Error",
                    "error": None,
                    "threshold": round(threshold, 8),
                })
        return jsonify({"total": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in browser")
    app.run(host="127.0.0.1", port=8000, debug=False)
"""

README_TXT = """EasyDefect Download Bundle
========================

This package includes:
1) OpenVINO model files for inference
2) A small local web app: welcome home + offline testing page

How to run (no coding):
-----------------------
1. Extract this zip to any folder.
2. Double-click: Open_Testing_Page.vbs  (or Open_Testing_Page.bat)
3. Your browser opens at the welcome page. Click "Go to testing".
4. Pick images in one list and click "Run test". Under each image you will see
   "Detected as: Normal" or "Detected as: Defective".

Offline usage:
--------------
- Runs locally with Python + OpenVINO (see run_local_server.py).
- If Python is not installed, install it once, then run the launcher again.
"""

RUN_BAT = """@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8000
python run_local_server.py
pause
"""

OPEN_TESTING_PAGE_BAT = """@echo off
cd /d "%~dp0"
set "_PY="
where py >nul 2>nul && set "_PY=py -3"
if not defined _PY (
    where python >nul 2>nul && set "_PY=python"
)
if not defined _PY (
    msg * "Python is not installed. Please contact provider."
    exit /b 1
)
start "" http://127.0.0.1:8000
%_PY% run_local_server.py
"""

OPEN_TESTING_PAGE_VBS = '''Set oShell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
cmd = "cmd /c cd /d """ & scriptDir & """ && Open_Testing_Page.bat"
oShell.Run cmd, 0, False
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


def _export_openvino_model(model_path, export_dir):
    try:
        from torchvision.models import resnet50, ResNet50_Weights
    except Exception as exc:
        raise RuntimeError(
            "torchvision is missing in backend environment. "
            "Install with: pip install torchvision"
        ) from exc

    convert_model = None
    save_model = None
    serialize = None

    # OpenVINO API differs across versions. Support both modern and legacy imports.
    try:
        from openvino import convert_model as ov_convert_model, save_model as ov_save_model
        convert_model = ov_convert_model
        save_model = ov_save_model
    except Exception:
        try:
            from openvino.tools.ovc import convert_model as ov_convert_model
            from openvino.runtime import serialize as ov_serialize
            convert_model = ov_convert_model
            serialize = ov_serialize
        except Exception as exc:
            raise RuntimeError(
                "OpenVINO is missing or incomplete in backend environment. "
                "Install with: pip install openvino openvino-dev"
            ) from exc

    device = torch.device("cpu")

    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
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
    encoder_xml = os.path.join(export_dir, "encoder.xml")
    autoencoder_xml = os.path.join(export_dir, "autoencoder.xml")
    if save_model is not None:
        save_model(encoder_ov, encoder_xml)
        save_model(autoencoder_ov, autoencoder_xml)
    else:
        serialize(encoder_ov, encoder_xml)
        serialize(autoencoder_ov, autoencoder_xml)
    shutil.copy2(os.path.join(model_path, "threshold.joblib"), os.path.join(export_dir, "threshold.joblib"))


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
    - Standalone testing page + local server files
    """
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
    bundle_dir = os.path.join(tmp_root, f"{model_name}_bundle")
    model_export_dir = os.path.join(bundle_dir, "model")
    os.makedirs(bundle_dir, exist_ok=True)

    try:
        _export_openvino_model(model_path, model_export_dir)

        display_name = (request.args.get("display_name") or "").strip()
        if not display_name:
            display_name = get_registered_user_display_name(email)
        if not display_name and email and "@" in email:
            display_name = email.split("@", 1)[0].strip()
        if not display_name:
            display_name = "User"

        home_html = OFFLINE_HOME_HTML.replace("__DISPLAY_NAME__", escape(display_name))

        with open(os.path.join(bundle_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(home_html)
        with open(os.path.join(bundle_dir, "test.html"), "w", encoding="utf-8") as f:
            f.write(OFFLINE_TEST_HTML)
        with open(os.path.join(bundle_dir, "run_local_server.py"), "w", encoding="utf-8") as f:
            f.write(RUN_SERVER_PY)
        with open(os.path.join(bundle_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(README_TXT)
        with open(os.path.join(bundle_dir, "start_test_server.bat"), "w", encoding="utf-8") as f:
            f.write(RUN_BAT)
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

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{model_name}_openvino_bundle.zip"
    )
