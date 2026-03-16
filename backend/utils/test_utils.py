import os
import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
import tempfile
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
_cache    = {}


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 512),       nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def extract_feature(img_path, feature_model, device):
    """L2-normalized ResNet50 feature — identical to train_utils."""
    img    = Image.open(img_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_model(tensor).squeeze().cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def load_artifacts(model_name: str) -> dict:
    if model_name in _cache:
        return _cache[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at: {model_path}")

    required = ["encoder.pth", "autoencoder.pth", "threshold.joblib"]
    missing  = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}. Retrain the model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Loading '{model_name}' on {device}")

    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.load_state_dict(
        torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    )
    feature_model.to(device).eval()

    autoencoder = Autoencoder(input_dim=2048)
    autoencoder.load_state_dict(
        torch.load(os.path.join(model_path, "autoencoder.pth"), map_location=device)
    )
    autoencoder.to(device).eval()

    threshold = float(joblib.load(os.path.join(model_path, "threshold.joblib")))
    print(f"[test] Threshold: {threshold:.8f}")

    arts = {
        "feature_model": feature_model,
        "autoencoder":   autoencoder,
        "threshold":     threshold,
        "device":        device,
    }
    _cache[model_name] = arts
    return arts


def predict_from_path(image_path: str, arts: dict) -> dict:
    device        = arts["device"]
    feature_model = arts["feature_model"]
    autoencoder   = arts["autoencoder"]
    threshold     = arts["threshold"]

    feat = extract_feature(image_path, feature_model, device)   # (2048,) L2-norm
    inp  = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out   = autoencoder(inp)
        error = float(torch.mean((out - inp) ** 2).item())

    label = "Normal" if error <= threshold else "Defective"
    return {"prediction": label, "error": round(error, 8), "threshold": round(threshold, 8)}


def test_images(model_name: str, normal_files: list, defective_files: list) -> dict:
    arts      = load_artifacts(model_name)
    threshold = arts["threshold"]
    tmp_dir   = tempfile.mkdtemp(prefix="easydefect_test_")

    print(f"\n[test] Model: {model_name}  Threshold: {threshold:.8f}")

    results = []
    correct = 0
    total   = 0

    def process(file_list, ground_truth):
        nonlocal correct, total
        for f in file_list:
            if not f or f.filename == "":
                continue
            tmp_path = os.path.join(tmp_dir, secure_filename(f.filename))
            try:
                f.stream.seek(0)
                f.save(tmp_path)
                if os.path.getsize(tmp_path) == 0:
                    raise ValueError("0 byte file")

                result     = predict_from_path(tmp_path, arts)
                is_correct = result["prediction"] == ground_truth
                if is_correct:
                    correct += 1
                total += 1

                print(
                    f"  {'OK   ' if is_correct else 'WRONG'} {f.filename:<28} "
                    f"error={result['error']:.8f}  thresh={threshold:.8f}  "
                    f"-> {result['prediction']} (actual={ground_truth})"
                )
                results.append({
                    "filename": f.filename, "prediction": result["prediction"],
                    "actual": ground_truth, "error": result["error"],
                    "threshold": result["threshold"], "correct": is_correct,
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({
                    "filename": f.filename, "prediction": "Error",
                    "actual": ground_truth, "error": None, "correct": False,
                })
                total += 1
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    process(normal_files,    "Normal")
    process(defective_files, "Defective")
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0
    print(f"[test] Result: {correct}/{total} correct ({accuracy}%)\n")
    return {
        "model_name": model_name, "accuracy": accuracy,
        "correct": correct, "total": total, "results": results,
    }