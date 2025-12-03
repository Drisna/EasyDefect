import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import argparse

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to load images and extract features
def load_images_and_features(dataset_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
    
    features = []
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(device)
            feature = model(img).squeeze().cpu().numpy()
            features.append(feature)
    return np.array(features)

# Main training function
def train_anomaly_detector(dataset_path, model_save_path, epochs=50, threshold_percentile=95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ResNet50 for features
    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.to(device)
    feature_model.eval()
    
    # Load features from training images
    print("Loading and extracting features from:", dataset_path)
    features = load_images_and_features(dataset_path, feature_model, device)
    if len(features) < 20:
        raise ValueError("At least 20 normal images required.")
    print(f"Loaded {len(features)} images.")
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Train Autoencoder
    autoencoder = Autoencoder(input_dim=scaled_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        inputs = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Compute reconstruction errors on training data
    with torch.no_grad():
        train_outputs = autoencoder(torch.tensor(scaled_features, dtype=torch.float32).to(device))
        train_errors = torch.mean((train_outputs - torch.tensor(scaled_features, dtype=torch.float32).to(device))**2, dim=1).cpu().numpy()
    
    # Set threshold as percentile of training errors
    threshold = np.percentile(train_errors, threshold_percentile)
    print(f"Threshold set at {threshold:.4f} (percentile: {threshold_percentile})")
    
    # Save models and data
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(feature_model.state_dict(), os.path.join(model_save_path, 'encoder.pth'))
    torch.save(autoencoder.state_dict(), os.path.join(model_save_path, 'autoencoder.pth'))
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler.joblib'))
    joblib.dump(threshold, os.path.join(model_save_path, 'threshold.joblib'))
    
    print("Training complete. Model saved to:", model_save_path)

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder Anomaly Detection Model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to folder with normal images")
    parser.add_argument('--model_save_path', type=str, required=True, help="Path to save trained model")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--threshold_percentile', type=int, default=95, help="Percentile for anomaly threshold")
    args = parser.parse_args()
    
    train_anomaly_detector(args.dataset_path, args.model_save_path, args.epochs, args.threshold_percentile)