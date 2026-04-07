import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import joblib
from PIL import Image

# ── Fix random seed so every training run gives identical results ─────────────
# Without this, weight initialization differs every run → different accuracy
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)


TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128),       nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, 512),       nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def extract_feature(img_path, feature_model, device):
    """
    L2-normalized ResNet50 feature.
    NO StandardScaler — it produces 10000x different values when
    transforming one image vs a batch, causing huge errors at test time.
    L2 normalization is stable for any batch size.
    """
    img    = Image.open(img_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_model(tensor).squeeze().cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def train_anomaly_detector(dataset_path, model_save_path, epochs=300):
    # ── Set seed FIRST — before any model or tensor creation ─────────────────
    set_seed(SEED)
    print(f"[train] Random seed fixed to {SEED} — results will be identical every run")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.to(device).eval()

    image_files = sorted([
        f for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    print(f"[train] Found {len(image_files)} training images")

    if len(image_files) < 20:
        raise ValueError(f"Need at least 20 images. Found: {len(image_files)}")

    image_paths = [os.path.join(dataset_path, f) for f in image_files]

    # Extract L2-normalized features
    features = []
    for p in image_paths:
        try:
            features.append(extract_feature(p, feature_model, device))
        except Exception as e:
            print(f"  Skip {os.path.basename(p)}: {e}")

    if len(features) < 20:
        raise ValueError(f"Need at least 20 valid images after preprocessing. Found: {len(features)}")

    features_np = np.array(features)
    print(f"[train] Feature shape: {features_np.shape}")
    print(f"[train] Feature range: [{features_np.min():.4f}, {features_np.max():.4f}]")

    inputs_t = torch.tensor(features_np, dtype=torch.float32).to(device)

    # Autoencoder — seed already set so weights init identically every run
    autoencoder = Autoencoder(input_dim=2048).to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion   = nn.MSELoss()

    print(f"[train] Training for {epochs} epochs...")
    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        loss = criterion(autoencoder(inputs_t), inputs_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={loss.item():.8f}")

    # ── Compute threshold via inference pipeline ──────────────────────────────
    autoencoder.eval()
    print(f"\n[train] Computing inference errors on training images...")
    inference_errors = []

    for p in image_paths:
        feat = extract_feature(p, feature_model, device)
        inp  = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out   = autoencoder(inp)
            error = float(torch.mean((out - inp) ** 2).item())
        inference_errors.append(error)
        print(f"  {os.path.basename(p):<20} error = {error:.8f}")

    errors = np.array(inference_errors)
    print(f"\n[train] Training errors:")
    print(f"  min  = {errors.min():.8f}")
    print(f"  mean = {errors.mean():.8f}")
    print(f"  max  = {errors.max():.8f}")

    # Robust thresholding to reduce under/over-classification.
    # max*3 can be too loose and often predicts everything as Normal.
    p99 = float(np.percentile(errors, 99))
    iqr = float(np.percentile(errors, 75) - np.percentile(errors, 25))
    threshold = p99 + 0.5 * max(iqr, 1e-8)
    print(f"\n[train] Threshold = p99 ({p99:.8f}) + 0.5*IQR ({iqr:.8f}) = {threshold:.8f}")
    print(f"[train] This threshold is now FIXED — retraining gives identical results")

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(feature_model.state_dict(),  os.path.join(model_save_path, 'encoder.pth'))
    torch.save(autoencoder.state_dict(),    os.path.join(model_save_path, 'autoencoder.pth'))
    joblib.dump(None,      os.path.join(model_save_path, 'scaler.joblib'))
    joblib.dump(threshold, os.path.join(model_save_path, 'threshold.joblib'))

    print(f"[train] Saved to: {model_save_path}\n")
    return model_save_path