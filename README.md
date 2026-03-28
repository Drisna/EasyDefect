# 🤖 EasyDefect — AI-Powered Anomaly Detection System

## 📌 Overview
EasyDefect is an offline AI-based product defect detection 
system using Transfer Learning (ResNet50) and Autoencoder 
for anomaly detection — deployable without internet.

## 🛠️ Tech Stack
- **Frontend:** React.js
- **Backend:** Flask (Python)
- **AI Model:** ResNet50 + Autoencoder (PyTorch)
- **Deployment:** OpenVINO (Intel CPU optimized)

## 🚀 How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend (Development)
```bash
cd new_frontend
npm install
npm start
```

### Production (Offline)
```bash
cd new_frontend
npm run build
cd ../backend
python app.py
# Open: http://localhost:5000
```

## 📋 How It Works
1. Upload 25+ normal product images
2. Train the anomaly detection model
3. Test with normal + defective images
4. Get predictions: NORMAL ✅ or DEFECTIVE ⚠️

## 📦 Offline Deployment
- Run `python app.py`
- Access via `http://localhost:5000`
- Works completely offline!
