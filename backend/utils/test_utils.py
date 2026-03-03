import os
import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Identical transform to train_utils.py
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model_cache = {}


def load_artifacts(model_name: str) -> dict:
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at: {model_path}")

    required = ["encoder.pth", "autoencoder.pth", "scaler.joblib", "threshold.joblib"]
    missing  = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Loading model '{model_name}' on {device}")

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

    scaler    = joblib.load(os.path.join(model_path, "scaler.joblib"))
    threshold = float(joblib.load(os.path.join(model_path, "threshold.joblib")))

    print(f"[test] Threshold loaded: {threshold:.8f}")

    artifacts = {
        "feature_model": feature_model,
        "autoencoder":   autoencoder,
        "scaler":        scaler,
        "threshold":     threshold,
        "device":        device,
    }
    _model_cache[model_name] = artifacts
    return artifacts


def predict_single(pil_image, artifacts: dict) -> dict:
    device        = artifacts["device"]
    feature_model = artifacts["feature_model"]
    autoencoder   = artifacts["autoencoder"]
    scaler        = artifacts["scaler"]
    threshold     = artifacts["threshold"]

    img_tensor = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: ResNet50 features
        features = feature_model(img_tensor).squeeze().cpu().numpy()

        # Step 2: Scale — same scaler fitted on training data
        scaled = scaler.transform([features])  # shape (1, 2048)
        scaled = scaled.clip(-10, 10)

        # Step 3: Reconstruct
        inp   = torch.tensor(scaled, dtype=torch.float32).to(device)
        recon = autoencoder(inp)
        error = float(torch.mean((recon - inp) ** 2).item())

    # ── Same logic as your working standalone script ─────────────────────────
    # Your standalone used: "Normal" if error <= abs(effective_threshold)
    # We do the same here. abs() handles any edge case where threshold
    # was saved as negative.
    prediction = "Normal" if error <= threshold else "Defective"

    return {
        "prediction": prediction,
        "error":      round(error, 6),
        "threshold":  round(abs(threshold), 6),
    }


def test_images(model_name: str, normal_files: list, defective_files: list) -> dict:
    artifacts = load_artifacts(model_name)
    threshold = artifacts["threshold"]

    results = []
    correct = 0
    total   = 0

    def process(file_list, ground_truth: str):
        nonlocal correct, total
        for f in file_list:
            if not f or f.filename == "":
                continue
            try:
                f.stream.seek(0)                          # reset stream — critical for Werkzeug
                pil_img = Image.open(f.stream).convert("RGB")
                f.stream.seek(0)

                result     = predict_single(pil_img, artifacts)
                is_correct = (result["prediction"] == ground_truth)

                if is_correct:
                    correct += 1
                total += 1

                print(
                    f"  {f.filename:<35} "
                    f"error={result['error']:.6f}  "
                    f"threshold={threshold:.6f}  "
                    f"-> {result['prediction']}  "
                    f"(actual={ground_truth})  "
                    f"{'OK' if is_correct else 'WRONG'}"
                )

                results.append({
                    "filename":   f.filename,
                    "prediction": result["prediction"],
                    "actual":     ground_truth,
                    "error":      result["error"],
                    "threshold":  round(abs(threshold), 6),
                    "correct":    is_correct,
                })

            except Exception as e:
                import traceback
                print(f"  ERROR on {f.filename}: {e}")
                traceback.print_exc()
                results.append({
                    "filename":   f.filename,
                    "prediction": "Error",
                    "actual":     ground_truth,
                    "error":      None,
                    "correct":    False,
                })
                total += 1

    print(f"\n[test] model={model_name}  threshold={abs(threshold):.8f}")
    process(normal_files,    "Normal")
    process(defective_files, "Defective")
    print(f"[test] Result: {correct}/{total} correct\n")

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0

    return {
        "model_name": model_name,
        "accuracy":   accuracy,
        "correct":    correct,
        "total":      total,
        "results":    results,
    }