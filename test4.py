import os
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
from PIL import Image
import argparse

# Function to load the trained model components
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MobileNetV2 with weights=None (no pretrained weights, since we're loading our own)
    model = mobilenet_v2(weights=None)
    model.classifier = torch.nn.Identity()
    model.load_state_dict(torch.load(os.path.join(model_path, 'encoder.pth'), map_location=device))
    model.to(device)
    model.eval()
    
    # Load scaler and OCSVM
    scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    ocsvm = joblib.load(os.path.join(model_path, 'ocsvm.joblib'))
    
    return model, scaler, ocsvm, device

# Function to preprocess and predict on a single image
def predict_anomaly(model, scaler, ocsvm, device, image_path):
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
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim
    
    # Extract features
    with torch.no_grad():
        features = model(img_tensor).squeeze().cpu().numpy()
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Predict with OCSVM
    prediction = ocsvm.predict(scaled_features)[0]  # 1 for normal, -1 for anomaly
    score = ocsvm.decision_function(scaled_features)[0]  # Higher score = more normal
    
    # Interpret
    label = "Normal" if prediction == 1 else "Anomaly"
    return label, score

# Function to test on a folder of images
def test_on_folder(model_path, test_folder_path):
    if not os.path.isdir(test_folder_path):
        raise ValueError(f"Invalid folder path: {test_folder_path}. Must be a directory.")
    
    model, scaler, ocsvm, device = load_model(model_path)
    
    results = []
    for filename in os.listdir(test_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(test_folder_path, filename)
            label, score = predict_anomaly(model, scaler, ocsvm, device, img_path)
            results.append(f"{filename}: {label} (Score: {score:.4f})")
            print(f"{filename}: {label} (Score: {score:.4f})")
    
    return results

# Main script for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anomaly Detection Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to folder with saved model files (encoder.pth, etc.)")
    parser.add_argument('--test_image', type=str, help="Path to a single test image file")
    parser.add_argument('--test_folder', type=str, help="Path to folder with test images")
    args = parser.parse_args()
    
    if args.test_image:
        model, scaler, ocsvm, device = load_model(args.model_path)
        label, score = predict_anomaly(model, scaler, ocsvm, device, args.test_image)
        print(f"Prediction for {args.test_image}: {label} (Score: {score:.4f})")
    elif args.test_folder:
        test_on_folder(args.model_path, args.test_folder)
    else:
        print("Provide either --test_image or --test_folder")