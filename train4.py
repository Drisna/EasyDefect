import os
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
from PIL import Image
import argparse

# Function to load and preprocess images
def load_images_from_folder(folder_path, transform):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
    return images

# Function to extract features using MobileNetV2
def extract_features(model, images, device):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            feature = model(img)  # Output: [1, 1280]
            features.append(feature.squeeze().cpu().numpy())
    return np.array(features)

# Main training function
def train_anomaly_detector(dataset_path, model_save_path, nu=0.1, gamma='scale'):
    """
    Trains an anomaly detection model using only normal images.
    
    Args:
        dataset_path (str): Path to folder containing only normal (good) images.
        model_save_path (str): Path to save the trained model files (encoder.pth, scaler.joblib, ocsvm.joblib).
        nu (float): OneClassSVM parameter (fraction of outliers, default 0.1).
        gamma (str or float): OneClassSVM RBF kernel parameter (default 'scale').
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained MobileNetV2 and modify for feature extraction
    model = mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()  # Remove classifier to get 1280-dim features
    model.to(device)
    
    # Define image preprocessing (ImageNet standard)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset (only normal images)
    print("Loading images from:", dataset_path)
    images = load_images_from_folder(dataset_path, transform)
    if len(images) < 20:
        raise ValueError("At least 20 normal images are required for training.")
    print(f"Loaded {len(images)} images.")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(model, images, device)
    print(f"Features shape: {features.shape}")
    
    # Normalize features with StandardScaler
    print("Normalizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Train One-Class SVM
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    ocsvm.fit(scaled_features)
    
    # Save the model components
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'encoder.pth'))
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler.joblib'))
    joblib.dump(ocsvm, os.path.join(model_save_path, 'ocsvm.joblib'))
    
    print("Training complete. Model saved to:", model_save_path)

# Example usage (can be called from Flask backend)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Detection Model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to folder with normal images")
    parser.add_argument('--model_save_path', type=str, required=True, help="Path to save trained model")
    parser.add_argument('--nu', type=float, default=0.1, help="OneClassSVM nu parameter")
    parser.add_argument('--gamma', type=str, default='scale', help="OneClassSVM gamma parameter")
    args = parser.parse_args()
    
    train_anomaly_detector(args.dataset_path, args.model_save_path, args.nu, args.gamma)
