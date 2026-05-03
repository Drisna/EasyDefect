import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TrainingPage.css";
import { getAuthHeaders, getCurrentUser, getCurrentUserDisplayName } from "../utils/auth";

const TrainingPage = () => {
  const [images,         setImages]         = useState([]);
  const [modelName,      setModelName]      = useState("");
  const [trainedModel,   setTrainedModel]   = useState("");
  const [trainingStatus, setTrainingStatus] = useState("idle");
  const [errorMsg,       setErrorMsg]       = useState("");

  const navigate = useNavigate();

  // ── KEY FIX: Clear uploads folder when page loads ────────────────────────
  // This prevents stale images from a previous session being included
  // in a new training run, which caused the "all defective" problem.
  useEffect(() => {
    fetch("http://localhost:5000/api/predict/clear", {
      method: "DELETE",
      headers: getAuthHeaders(),
    })
      .then(res => res.json())
      .then(data => console.log("[TrainingPage] Cleared uploads:", data.message))
      .catch(err => console.warn("[TrainingPage] Could not clear uploads:", err));
  }, []);


  // ── Upload each file to backend as it is selected ────────────────────────
  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setErrorMsg("");

    const newImages = files.map(file => ({
      file,
      url:      URL.createObjectURL(file),
      uploaded: false,
      message:  "Uploading…",
    }));

    setImages(prev => [...prev, ...newImages]);
    setTrainingStatus("uploading");

    for (const img of newImages) {
      const formData = new FormData();
      formData.append("file", img.file);

      try {
        const res = await fetch("http://localhost:5000/api/predict/", {
          method: "POST",
          headers: getAuthHeaders(),
          body:   formData,
        });

        let data = {};
        try { data = await res.json(); }
        catch { data = { success: false, message: `Server error ${res.status}` }; }

        setImages(prev =>
          prev.map(item =>
            item.file === img.file
              ? {
                  ...item,
                  uploaded: data.success,
                  message:  data.success ? "✅ Uploaded" : `❌ ${data.message}`
                }
              : item
          )
        );
      } catch (err) {
        setImages(prev =>
          prev.map(item =>
            item.file === img.file
              ? { ...item, uploaded: false, message: `❌ ${err.message}` }
              : item
          )
        );
      }
    }

    setTrainingStatus("idle");
  };


  const removeImage = (index) => {
    setImages(images.filter((_, i) => i !== index));
  };


  // ── Start training ────────────────────────────────────────────────────────
  const handleTrain = async () => {
    setErrorMsg("");

    if (!modelName.trim()) {
      setErrorMsg("Please enter a model name.");
      return;
    }
    if (images.length < 20) {
      setErrorMsg(`Minimum 20 images required. You have ${images.length}.`);
      return;
    }

    const failed = images.filter(img => !img.uploaded);
    if (failed.length > 0) {
      setErrorMsg(`${failed.length} image(s) failed to upload. Remove them and try again.`);
      return;
    }

    setTrainingStatus("training");

    try {
      const response = await fetch("http://localhost:5000/api/train/", {
        method:  "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body:    JSON.stringify({ model_name: modelName.trim(), epochs: 100 }),
      });

      let data = {};
      try { data = await response.json(); }
      catch { data = { success: false, message: `Server error ${response.status}` }; }

      if (data.success) {
        setTrainedModel(modelName.trim());
        setTrainingStatus("trained");
        // Clear uploads after successful training
        fetch("http://localhost:5000/api/predict/clear", {
          method: "DELETE",
          headers: getAuthHeaders(),
        })
          .catch(() => {});
      } else {
        setTrainingStatus("idle");
        setErrorMsg(`Training failed: ${data.message}`);
      }
    } catch (err) {
      setTrainingStatus("idle");
      setErrorMsg(`Server error: ${err.message}. Is the backend running?`);
    }
  };

  const handleDownload = () => {
    const nameToDownload = trainedModel || modelName.trim();
    if (!nameToDownload) return;

    const email = encodeURIComponent(getCurrentUser() || "");
    const displayName = encodeURIComponent(getCurrentUserDisplayName() || "");
    const encodedModel = encodeURIComponent(nameToDownload);
    window.location.href =
      `http://localhost:5000/api/models/download/${encodedModel}?user_email=${email}&display_name=${displayName}`;
  };


  const uploadedCount = images.filter(i => i.uploaded).length;

  return (
    <div className="training-page">
      <h1>Train Your Model</h1>

      <div className="training-container">

        {/* LEFT: Image preview */}
        <div className="image-panel">
          <h3>Training Images</h3>
          <p className="count-text">
            {uploadedCount} / {images.length} uploaded
            {images.length === 0     ? "" :
             images.length >= 20     ? " ✅ Ready to train" :
             ` — need ${20 - images.length} more`}
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
                >✕</button>
                <p className={`upload-status ${img.uploaded ? "success-text" : "error-text"}`}>
                  {img.message}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT: Training controls */}
        <div className="train-panel">
          <div className="train-center">
            <h3>Training Setup</h3>

            <input
              type="text"
              placeholder="Enter Model Name (e.g. PCB_Model)"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              disabled={trainingStatus === "training"}
            />

            {errorMsg && (
              <p className="error-text" style={{ marginTop: "10px" }}>
                {errorMsg}
              </p>
            )}

            {(trainingStatus === "idle" || trainingStatus === "uploading") && (
              <button
                className="btn primary"
                onClick={handleTrain}
                disabled={trainingStatus === "uploading"}
              >
                {trainingStatus === "uploading" ? "Uploading images…" : "Start Training"}
              </button>
            )}

            {trainingStatus === "training" && (
              <p className="note">⏳ Training in progress… This may take a few minutes.</p>
            )}

            {trainingStatus === "trained" && (
              <>
                <p className="success-text">✅ Training Complete!</p>
                <button className="btn primary" onClick={handleDownload}>
                  Download Offline App
                </button>
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
