# import os
# import torch
# import torch.nn as nn
# import joblib
# import numpy as np

# from torchvision import transforms
# from torchvision.models import resnet50
# from PIL import Image

# from models.autoencoder import Autoencoder

# MODEL_DIR = "models"


# def test_images(model_name, normal_files, defective_files):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_path = os.path.join(MODEL_DIR, model_name)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model '{model_name}' not found")

#     # -----------------------------
#     # Load Encoder (ResNet50)
#     # -----------------------------
#     feature_model = resnet50()
#     feature_model.fc = nn.Identity()

#     feature_model.load_state_dict(
#         torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
#     )

#     feature_model.to(device)
#     feature_model.eval()

#     # -----------------------------
#     # Load Autoencoder
#     # -----------------------------
#     autoencoder = Autoencoder()

#     autoencoder.load_state_dict(
#         torch.load(os.path.join(model_path, "autoencoder.pth"), map_location=device)
#     )

#     autoencoder.to(device)
#     autoencoder.eval()

#     # -----------------------------
#     # Load Scaler & Threshold
#     # -----------------------------
#     scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
#     threshold = joblib.load(os.path.join(model_path, "threshold.joblib"))

#     # -----------------------------
#     # Image Transform
#     # -----------------------------
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         )
#     ])

#     results = []
#     correct = 0
#     total = 0

#     def process_files(files, ground_truth):
#         nonlocal correct, total

#         for file in files:
#             img = Image.open(file).convert("RGB")
#             img = transform(img).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 features = feature_model(img).cpu().numpy()
#                 features = scaler.transform(features)

#                 tensor = torch.tensor(features, dtype=torch.float32).to(device)
#                 reconstruction = autoencoder(tensor)

#                 error = torch.mean((reconstruction - tensor) ** 2).item()

#             prediction = "Defective" if error > threshold else "Normal"

#             if prediction == ground_truth:
#                 correct += 1

#             total += 1

#             results.append({
#                 "filename": file.filename,
#                 "prediction": prediction,
#                 "actual": ground_truth,
#                 "error": round(error, 6)
#             })

#     # Process both categories
#     process_files(normal_files, "Normal")
#     process_files(defective_files, "Defective")

#     accuracy = (correct / total) * 100 if total > 0 else 0

#     return {
#         "accuracy": round(accuracy, 2),
#         "correct": correct,
#         "total": total,
#         "results": results
#     }

import os
import torch
import torch.nn as nn
import joblib
import numpy as np

from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from torchvision.models import ResNet50_Weights


from models.autoencoder import Autoencoder

MODEL_DIR = "models"


def validate_model_files(model_path):
    required_files = [
        "encoder.pth",
        "autoencoder.pth",
        "scaler.joblib",
        "threshold.joblib"
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            raise FileNotFoundError(f"{file} missing in model folder")


def test_images(model_name, normal_files, defective_files):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found")

    validate_model_files(model_path)

    print("=== LOADING MODEL ===")
    print("Model:", model_name)
    print("Device:", device)

    # -----------------------------
    # Load Encoder (ResNet50)
    # -----------------------------
   # feature_model = resnet50(weights=None)  # important
    feature_model = resnet50(weights=ResNet50_Weights.DEFAULT)

    feature_model.fc = nn.Identity()

    feature_model.load_state_dict(
        torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    )

    feature_model.to(device)
    feature_model.eval()

    print("✅ Encoder loaded")

    # -----------------------------
    # Load Autoencoder
    # -----------------------------
    autoencoder = Autoencoder()

    autoencoder.load_state_dict(
        torch.load(os.path.join(model_path, "autoencoder.pth"), map_location=device)
    )

    autoencoder.to(device)
    autoencoder.eval()

    print("✅ Autoencoder loaded")

    # -----------------------------
    # Load Scaler & Threshold
    # -----------------------------
    scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
    threshold = joblib.load(os.path.join(model_path, "threshold.joblib"))

    print("✅ Scaler loaded")
    print("✅ Threshold loaded:", threshold)
    print("======================")

    # -----------------------------
    # Image Transform
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    results = []
    correct = 0
    total = 0

    def process_files(files, ground_truth):
        nonlocal correct, total

        for file in files:
            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = feature_model(img).cpu().numpy()
                features = scaler.transform(features)

                tensor = torch.tensor(features, dtype=torch.float32).to(device)
                reconstruction = autoencoder(tensor)

                error = torch.mean((reconstruction - tensor) ** 2).item()

            prediction = "Defective" if error > threshold else "Normal"

            print(f"{file.filename} → Error: {error:.6f} → {prediction}")

            if prediction == ground_truth:
                correct += 1

            total += 1

            results.append({
                "filename": file.filename,
                "prediction": prediction,
                "actual": ground_truth,
                "error": round(error, 6)
            })

    process_files(normal_files, "Normal")
    process_files(defective_files, "Defective")

    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "results": results
    }
