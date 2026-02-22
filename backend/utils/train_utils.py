import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image


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


def load_images_and_features(dataset_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = [
        f for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    print(f"  Found {len(image_files)} images in {dataset_path}")

    features = []
    for filename in image_files:
        img_path = os.path.join(dataset_path, filename)
        try:
            img  = Image.open(img_path).convert('RGB')
            img  = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img).squeeze().cpu().numpy()
            features.append(feat)
        except Exception as e:
            print(f"  Skipping {filename}: {e}")

    if not features:
        raise ValueError(f"No valid images in: {dataset_path}")

    return np.array(features)


def train_anomaly_detector(dataset_path, model_save_path, epochs=100, threshold_percentile=95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[train] Device: {device}")

    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.to(device).eval()

    features = load_images_and_features(dataset_path, feature_model, device)

    if len(features) < 20:
        raise ValueError(f"Need at least 20 images. Found: {len(features)}")

    print(f"[train] Feature shape : {features.shape}")

    scaler          = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    autoencoder = Autoencoder(input_dim=scaled_features.shape[1]).to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion   = nn.MSELoss()

    inputs_t = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    print(f"[train] Training for {epochs} epochs...")
    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        loss = criterion(autoencoder(inputs_t), inputs_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={loss.item():.8f}")

    # ── Compute errors on ALL training images ────────────────────────────────
    autoencoder.eval()
    with torch.no_grad():
        recon  = autoencoder(inputs_t)
        errors = torch.mean((recon - inputs_t) ** 2, dim=1).cpu().numpy()

    print(f"\n[train] Training reconstruction errors:")
    print(f"  min  = {errors.min():.8f}")
    print(f"  mean = {errors.mean():.8f}")
    print(f"  max  = {errors.max():.8f}")
    print(f"  p95  = {np.percentile(errors, 95):.8f}")
    print(f"  p99  = {np.percentile(errors, 99):.8f}")

    # ── Threshold: same percentile as your working standalone script ─────────
    # We use p95 (same default as your original) but multiply by 3.0
    # to give test images room to vary slightly from training images.
    # Your standalone script effectively did this by using abs() on a
    # threshold that was computed differently — this multiplier achieves
    # the same effect explicitly and reliably.
    base_threshold = float(np.percentile(errors, threshold_percentile))
    threshold      = base_threshold * 3.0

    print(f"\n[train] base p{threshold_percentile} = {base_threshold:.8f}")
    print(f"[train] final threshold (x3.0) = {threshold:.8f}")

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(feature_model.state_dict(),  os.path.join(model_save_path, 'encoder.pth'))
    torch.save(autoencoder.state_dict(),    os.path.join(model_save_path, 'autoencoder.pth'))
    joblib.dump(scaler,    os.path.join(model_save_path, 'scaler.joblib'))
    joblib.dump(threshold, os.path.join(model_save_path, 'threshold.joblib'))

    print(f"[train] Model saved to: {model_save_path}\n")
    return model_save_path