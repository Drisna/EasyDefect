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

# Autoencoder class (must match training)
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

# Function to load the trained model components
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ResNet50 for features
    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.load_state_dict(torch.load(os.path.join(model_path, 'encoder.pth'), map_location=device))
    feature_model.to(device)
    feature_model.eval()
    
    # Load Autoencoder
    autoencoder = Autoencoder(input_dim=2048).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(model_path, 'autoencoder.pth'), map_location=device))
    autoencoder.eval()
    
    # Load scaler and threshold
    scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    threshold = joblib.load(os.path.join(model_path, 'threshold.joblib'))
    
    return feature_model, autoencoder, scaler, threshold, device

# Function to preprocess and predict on a single image
def predict_anomaly(feature_model, autoencoder, scaler, threshold, device, image_path, custom_threshold=None):
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}. Must be a file, not a folder.")
    
    # Define preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_model(img_tensor).squeeze().cpu().numpy()
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Reconstruct with autoencoder
    inputs = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = autoencoder(inputs)
    
    # Compute reconstruction error (MSE)
    error = torch.mean((outputs - inputs)**2).item()
    
    # Use custom threshold if provided, else saved threshold
    effective_threshold = custom_threshold if custom_threshold is not None else threshold
    
    # Predict based on threshold
    label = "Normal" if error <= abs(effective_threshold) else "Anomaly"  # abs for negative scores
    score = -error  # Negative score for anomaly
    
    return label, score

# Function to test on a folder of images
def test_on_folder(model_path, test_folder_path, custom_threshold=None):
    if not os.path.isdir(test_folder_path):
        raise ValueError(f"Invalid folder path: {test_folder_path}. Must be a directory.")
    
    feature_model, autoencoder, scaler, threshold, device = load_model(model_path)
    
    results = []
    for filename in os.listdir(test_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(test_folder_path, filename)
            label, score = predict_anomaly(feature_model, autoencoder, scaler, threshold, device, img_path, custom_threshold)
            results.append(f"{filename}: {label} (Score: {score:.4f})")
            print(f"{filename}: {label} (Score: {score:.4f})")
    
    return results

# Main script for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Autoencoder Anomaly Detection Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to folder with saved model files")
    parser.add_argument('--test_image', type=str, help="Path to a single test image file")
    parser.add_argument('--test_folder', type=str, help="Path to folder with test images")
    parser.add_argument('--custom_threshold', type=float, help="Custom threshold for anomaly detection (e.g., -5.0)")
    args = parser.parse_args()
    
    if args.test_image:
        feature_model, autoencoder, scaler, threshold, device = load_model(args.model_path)
        label, score = predict_anomaly(feature_model, autoencoder, scaler, threshold, device, args.test_image, args.custom_threshold)
        print(f"Prediction for {args.test_image}: {label} (Score: {score:.4f})")
    elif args.test_folder:
        test_on_folder(args.model_path, args.test_folder, args.custom_threshold)
    else:
        print("Provide either --test_image or --test_folder")