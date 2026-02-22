"""
Drop this file into your backend/ folder and run:
    python diagnose.py <model_name> <folder_with_test_images>

Example:
    python diagnose.py my_model test_images/
"""

import sys
import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def run(model_name, image_folder):
    base       = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "models", model_name)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*65)
    print(f"  Model  : {model_path}")
    print(f"  Images : {image_folder}")
    print(f"  Device : {device}")
    print("="*65 + "\n")

    scaler    = joblib.load(os.path.join(model_path, "scaler.joblib"))
    threshold = float(joblib.load(os.path.join(model_path, "threshold.joblib")))

    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.load_state_dict(
        torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    )
    feature_model.to(device).eval()

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(
        torch.load(os.path.join(model_path, "autoencoder.pth"), map_location=device)
    )
    autoencoder.to(device).eval()

    print(f"  Saved threshold = {threshold:.8f}\n")

    files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not files:
        print(f"No images found in {image_folder}")
        return

    print(f"{'Filename':<35} {'Error':>12}  {'Ratio':>8}  Result")
    print("-" * 70)

    errors = []
    for fname in sorted(files):
        fpath = os.path.join(image_folder, fname)
        img   = Image.open(fpath).convert("RGB")
        t     = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat   = feature_model(t).cpu().numpy()
            scaled = scaler.transform(feat)
            inp    = torch.tensor(scaled, dtype=torch.float32).to(device)
            recon  = autoencoder(inp)
            error  = float(torch.mean((recon - inp) ** 2).item())

        ratio = error / threshold
        label = "DEFECTIVE" if error > threshold else "NORMAL"
        errors.append(error)
        print(f"  {fname:<33} {error:>12.6f}  {ratio:>7.2f}x  {label}")

    errors = np.array(errors)
    print("\n" + "="*65)
    print(f"  Threshold             : {threshold:.8f}")
    print(f"  Error min             : {errors.min():.8f}")
    print(f"  Error mean            : {errors.mean():.8f}")
    print(f"  Error max             : {errors.max():.8f}")
    print(f"  Above threshold       : {(errors > threshold).sum()} / {len(errors)}")
    print(f"  Mean/Threshold ratio  : {errors.mean()/threshold:.2f}x")
    print("="*65 + "\n")

    ratio = errors.mean() / threshold
    if ratio > 10:
        print("DIAGNOSIS: Errors are 10x+ above threshold.")
        print("  Most likely cause: uploads/ folder had stale/wrong images during training.")
        print("  ACTION: Delete model folder, clear uploads/, retrain from scratch.")
    elif ratio > 2:
        print("DIAGNOSIS: Errors moderately above threshold.")
        print("  Test images differ from training images (lighting/zoom/background).")
        print("  ACTION: Use more varied training images and retrain.")
    elif ratio > 1:
        print("DIAGNOSIS: Errors just above threshold — threshold too strict.")
        print("  ACTION: Retrain with threshold_percentile=99 and 1.5x multiplier.")
    else:
        print("DIAGNOSIS: Errors BELOW threshold — should predict Normal.")
        print("  There is a bug in test_utils.py prediction logic.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose.py <model_name> <image_folder>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])