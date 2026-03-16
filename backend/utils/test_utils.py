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
_model_cache = {}


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x  = self.layer0(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        v2 = self.pool(f2).flatten(1)
        v3 = self.pool(f3).flatten(1)
        return torch.cat([v2, v3], dim=1)


def l2_normalize(feat):
    norm = np.linalg.norm(feat)
    return (feat / norm).astype(np.float32) if norm > 0 else feat.astype(np.float32)


def extract_feature(img_path, model, device):
    img    = Image.open(img_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    return l2_normalize(feat)


def load_artifacts(model_name: str) -> dict:
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at: {model_path}")

    if not os.path.exists(os.path.join(model_path, 'memory_bank.joblib')):
        raise FileNotFoundError(
            f"Old model format. Delete '{model_name}' folder and retrain."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Loading model '{model_name}' on {device}")

    model       = FeatureExtractor().to(device).eval()
    memory_bank = joblib.load(os.path.join(model_path, 'memory_bank.joblib'))
    threshold   = float(joblib.load(os.path.join(model_path, 'threshold.joblib')))

    print(f"[test] Memory bank : {memory_bank.shape}")
    print(f"[test] Threshold   : {threshold:.6f}")

    artifacts = {
        "model": model, "memory_bank": memory_bank,
        "threshold": threshold, "device": device,
    }
    _model_cache[model_name] = artifacts
    return artifacts


def predict_from_path(image_path, artifacts):
    feat  = extract_feature(image_path, artifacts["model"], artifacts["device"])
    dists = np.linalg.norm(artifacts["memory_bank"] - feat, axis=1)
    score = float(np.sort(dists)[:3].mean())  # k=3, matches training
    label = "Normal" if score <= artifacts["threshold"] else "Defective"
    return {"prediction": label, "error": round(score, 6),
            "threshold": round(artifacts["threshold"], 6)}


def test_images(model_name, normal_files, defective_files):
    artifacts = load_artifacts(model_name)
    threshold = artifacts["threshold"]
    tmp_dir   = tempfile.mkdtemp(prefix="easydefect_test_")

    print(f"\n[test] Model: {model_name}  Threshold: {threshold:.6f}")

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

                result     = predict_from_path(tmp_path, artifacts)
                is_correct = result["prediction"] == ground_truth
                if is_correct:
                    correct += 1
                total += 1

                print(
                    f"  {'OK   ' if is_correct else 'WRONG'} "
                    f"{f.filename:<28} score={result['error']:.6f}  "
                    f"thresh={threshold:.6f}  "
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

    process(normal_files, "Normal")
    process(defective_files, "Defective")
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0
    print(f"[test] Result: {correct}/{total} correct ({accuracy}%)\n")
    return {"model_name": model_name, "accuracy": accuracy,
            "correct": correct, "total": total, "results": results}