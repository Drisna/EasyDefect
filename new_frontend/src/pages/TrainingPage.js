import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TrainingPage.css";

const API = "http://localhost:5000";

const TrainingPage = () => {
  const [images, setImages]               = useState([]);
  const [modelName, setModelName]         = useState("");
  const [trainingStatus, setTrainingStatus] = useState("idle"); // idle | training | trained
  const navigate = useNavigate();

  // 🔥 FIX 1: Clear stale uploads on mount so old sessions don't pollute training
  useEffect(() => {
    fetch(`${API}/api/predict/clear`, { method: "DELETE" })
      .then(res => res.json())
      .then(data => console.log("[mount] Cleared uploads:", data.message))
      .catch(err => console.warn("[mount] Could not clear uploads:", err));
  }, []);

  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);

    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      uploaded: false,
      message: "Uploading...",
    }));

    setImages((prev) => [...prev, ...newImages]);

    // Upload each file to backend immediately
    for (const img of newImages) {
      const formData = new FormData();
      formData.append("file", img.file);

      try {
        const res  = await fetch(`${API}/api/predict/`, { method: "POST", body: formData });
        const data = await res.json();

        setImages((prev) =>
          prev.map((item) =>
            item.file === img.file
              ? { ...item, uploaded: data.success, message: data.message }
              : item
          )
        );
      } catch (err) {
        setImages((prev) =>
          prev.map((item) =>
            item.file === img.file
              ? { ...item, uploaded: false, message: "Upload failed" }
              : item
          )
        );
        console.error("Upload error:", err);
      }
    }
  };

  const removeImage = (index) => {
    setImages(images.filter((_, i) => i !== index));
  };

  const handleTrain = async () => {
    if (!modelName.trim()) {
      alert("Please enter a model name");
      return;
    }

    // 🔥 FIX 2: Backend requires 20, so check 20 (was checking 25 — inconsistent)
    if (images.length < 20) {
      alert("Minimum 20 images are required to train the model");
      return;
    }

    const notUploaded = images.filter((img) => !img.uploaded);
    if (notUploaded.length > 0) {
      alert(`${notUploaded.length} image(s) haven't finished uploading. Please wait.`);
      return;
    }

    setTrainingStatus("training");

    try {
      const response = await fetch(`${API}/api/train/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName, epochs: 50 }),
      });

      const data = await response.json();
      console.log("Training response:", data);

      if (data.success) {
        // 🔥 FIX 3: Clear uploads after training so they don't affect next session
        await fetch(`${API}/api/predict/clear`, { method: "DELETE" });
        setTrainingStatus("trained");
        alert(`✅ Training Complete: ${data.message}`);
      } else {
        setTrainingStatus("idle");
        alert(`❌ Training Failed: ${data.message}`);
      }
    } catch (err) {
      console.error("Error training model:", err);
      setTrainingStatus("idle");
      alert("Training failed due to server error. Is the backend running?");
    }
  };

  const uploadedCount = images.filter(i => i.uploaded).length;

  return (
    <div className="training-page">
      <h1>Train Your Model</h1>

      <div className="training-container">

        {/* LEFT — Image Preview */}
        <div className="image-panel">
          <h3>Training Images</h3>
          <p className="count-text">
            {images.length} selected &nbsp;|&nbsp; {uploadedCount} uploaded to server
          </p>

          <input
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.bmp"
            onChange={handleImageUpload}
            disabled={trainingStatus === "training"}
          />

          <div className="image-grid">
            {images.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="training" />
                <button
                  className="delete-btn"
                  onClick={() => removeImage(index)}
                  disabled={trainingStatus === "training"}
                >
                  ✕
                </button>
                <p className={`upload-status ${img.uploaded ? "success-text" : "error-text"}`}>
                  {img.message}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT — Training Controls */}
        <div className="train-panel">
          <div className="train-center">
            <h3>Training Setup</h3>

            <input
              type="text"
              placeholder="Enter Model Name"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={trainingStatus === "training"}
            />

            {trainingStatus === "idle" && (
              <button className="btn primary" onClick={handleTrain}>
                Start Training
              </button>
            )}

            {trainingStatus === "training" && (
              <div>
                <div className="spinner"></div>
                <p className="note">Training in progress… this may take a few minutes.</p>
              </div>
            )}

            {trainingStatus === "trained" && (
              <>
                <p className="success-text">✅ Training Complete!</p>
                <button className="btn secondary" onClick={() => navigate("/test")}>
                  Test Model →
                </button>
              </>
            )}

            <p className="note">Minimum 20 normal images required for training.</p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default TrainingPage;