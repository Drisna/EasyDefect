import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import joblib
from PIL import Image


# ── Transform — identical in train and test ──────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Autoencoder with very tight bottleneck ───────────────────────────────────
# Bottleneck = 64 (vs 256 before). Smaller bottleneck = harder to reconstruct
# anything the model hasn't seen. Defective images will fail to compress/decompress
# correctly because the bottleneck only learned the normal manifold.
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
    """
    Extract raw ResNet50 features and L2-normalize.
    NO StandardScaler — it causes 10000x errors on single-image transform.
    L2 normalization maps all features to unit sphere, stable for any batch size.
    """
    img    = Image.open(img_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_model(tensor).squeeze().cpu().numpy()  # (2048,)
    # L2 normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def train_anomaly_detector(dataset_path, model_save_path, epochs=300):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[train] Device: {device}")
    print(f"[train] Method: Autoencoder on L2-normalized ResNet50 features")
    print(f"[train] Bottleneck: 2048 -> 512 -> 128 -> 64 -> 128 -> 512 -> 2048")

    # Feature extractor (frozen)
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

    # Extract L2-normalized features for all training images
    features = []
    for p in image_paths:
        try:
            features.append(extract_feature(p, feature_model, device))
        except Exception as e:
            print(f"  Skip {os.path.basename(p)}: {e}")

    features_np = np.array(features)  # (N, 2048), all unit vectors
    print(f"[train] Feature shape : {features_np.shape}")
    print(f"[train] Feature range : [{features_np.min():.4f}, {features_np.max():.4f}]")

    inputs_t = torch.tensor(features_np, dtype=torch.float32).to(device)

    # Train autoencoder
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
    # Run each training image through extract_feature() then autoencoder
    # This is IDENTICAL to what test_utils does, so errors are calibrated
    autoencoder.eval()
    print(f"\n[train] Computing inference-pipeline errors on training images...")
    inference_errors = []

    for p in image_paths:
        feat = extract_feature(p, feature_model, device)            # L2-normalized (2048,)
        inp  = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out   = autoencoder(inp)
            error = float(torch.mean((out - inp) ** 2).item())
        inference_errors.append(error)
        print(f"  {os.path.basename(p):<20} error = {error:.8f}")

    errors = np.array(inference_errors)
    print(f"\n[train] Inference errors on training images:")
    print(f"  min  = {errors.min():.8f}")
    print(f"  mean = {errors.mean():.8f}")
    print(f"  max  = {errors.max():.8f}")
    print(f"  p95  = {np.percentile(errors, 95):.8f}")
    print(f"  p99  = {np.percentile(errors, 99):.8f}")

    # Threshold = max * 3.0
    # max guarantees all training images pass as Normal
    # 3.0x buffer for unseen normal test images
    threshold = float(errors.max()) * 3.0
    print(f"\n[train] Threshold = max ({errors.max():.8f}) x 3.0 = {threshold:.8f}")

    # Save
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(feature_model.state_dict(),  os.path.join(model_save_path, 'encoder.pth'))
    torch.save(autoencoder.state_dict(),    os.path.join(model_save_path, 'autoencoder.pth'))
    joblib.dump(None,      os.path.join(model_save_path, 'scaler.joblib'))   # no scaler
    joblib.dump(threshold, os.path.join(model_save_path, 'threshold.joblib'))

    print(f"[train] Saved to: {model_save_path}\n")
    return model_save_path