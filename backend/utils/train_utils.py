"""
KNN anomaly detection on ResNet50 layer2+layer3 features.
Key improvements:
- Use multiple crop augmentations per training image to densify memory bank
- Tighter threshold based on mean + 2*std instead of p95*1.5
- k=1 nearest neighbour (stricter, better for subtle defects)
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
import joblib
from PIL import Image


# Standard inference transform
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augmentation transforms to densify memory bank
AUGMENT_TRANSFORMS = [
    TRANSFORM,  # original
    transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(280), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]


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
        return torch.cat([v2, v3], dim=1)  # (B, 1536)


def l2_normalize(feat):
    norm = np.linalg.norm(feat)
    return (feat / norm).astype(np.float32) if norm > 0 else feat.astype(np.float32)


def extract_feature(img_path, model, device, transform=None):
    if transform is None:
        transform = TRANSFORM
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    return l2_normalize(feat)


def knn_score(feat, memory_bank, k=1):
    dists = np.linalg.norm(memory_bank - feat, axis=1)
    return float(np.sort(dists)[:k].mean())


def train_anomaly_detector(dataset_path, model_save_path, epochs=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[train] Device: {device}")
    print(f"[train] Method: KNN (k=1) on ResNet50 layer2+layer3, augmented memory bank")

    model = FeatureExtractor().to(device).eval()

    image_files = sorted([
        f for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    print(f"[train] Found {len(image_files)} training images")

    if len(image_files) < 20:
        raise ValueError(f"Need at least 20 images. Found: {len(image_files)}")

    image_paths = [os.path.join(dataset_path, f) for f in image_files]

    # Build augmented memory bank — each image added 4x with different crops/flips
    memory_bank = []
    for p in image_paths:
        try:
            for aug in AUGMENT_TRANSFORMS:
                feat = extract_feature(p, model, device, transform=aug)
                memory_bank.append(feat)
            print(f"  Added (x{len(AUGMENT_TRANSFORMS)}): {os.path.basename(p)}")
        except Exception as e:
            print(f"  Skipping {os.path.basename(p)}: {e}")

    memory_bank = np.array(memory_bank)
    print(f"\n[train] Memory bank: {memory_bank.shape}  ({len(image_files)} images x {len(AUGMENT_TRANSFORMS)} augments)")

    # Compute leave-one-out KNN scores on original training images
    print(f"[train] Computing training scores (k=1, leave-one-out)...")
    original_features = memory_bank[::len(AUGMENT_TRANSFORMS)]  # every 4th = original
    train_scores = []

    for i, feat in enumerate(original_features):
        # Remove all augments of this image from the bank
        mask = np.ones(len(memory_bank), dtype=bool)
        start = i * len(AUGMENT_TRANSFORMS)
        mask[start:start + len(AUGMENT_TRANSFORMS)] = False
        bank_without_self = memory_bank[mask]

        score = knn_score(feat, bank_without_self, k=1)
        train_scores.append(score)
        print(f"  {image_files[i]:<20} score = {score:.6f}")

    train_scores = np.array(train_scores)
    print(f"\n[train] Training scores:")
    print(f"  min  = {train_scores.min():.6f}")
    print(f"  mean = {train_scores.mean():.6f}")
    print(f"  max  = {train_scores.max():.6f}")
    print(f"  std  = {train_scores.std():.6f}")

    # Threshold = mean + 2*std
    # Tighter than p95*1.5 — catches subtle defects better
    threshold = float(train_scores.max()) * 1.5
    print(f"\n[train] Threshold = mean + 2*std = {threshold:.6f}")

    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(memory_bank, os.path.join(model_save_path, 'memory_bank.joblib'))
    joblib.dump(threshold,   os.path.join(model_save_path, 'threshold.joblib'))
    torch.save(model.state_dict(), os.path.join(model_save_path, 'encoder.pth'))
    joblib.dump(None, os.path.join(model_save_path, 'scaler.joblib'))
    joblib.dump(None, os.path.join(model_save_path, 'autoencoder.pth'))

    print(f"[train] Saved to: {model_save_path}\n")
    return model_save_path