import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TrainingPage.css";


const TrainingPage = () => {
  const [images, setImages] = useState([]);
  const [modelName, setModelName] = useState("");
  const [trainingStatus, setTrainingStatus] = useState("idle"); 
  // idle | training | trained

  const navigate = useNavigate();

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
    }));
    setImages((prev) => [...prev, ...newImages]);
  };

  const removeImage = (index) => {
    setImages(images.filter((_, i) => i !== index));
  };

  const handleTrain = () => {
    if (!modelName.trim()) {
      alert("Please enter a model name");
      return;
    }

    if (images.length < 25) {
      alert("Minimum 25 images are required to train the model");
      return;
    }

    setTrainingStatus("training");

    // Simulated training time
    setTimeout(() => {
      setTrainingStatus("trained");
    }, 3000);
  };

  return (
    <div className="training-page">
      <h1>Train Your Model</h1>

      <div className="training-container">
        {/* LEFT HALF – IMAGE PREVIEW */}
        <div className="image-panel">
          <h3>Training Images</h3>
          <p className="count-text">{images.length} / 25 images selected</p>

          <input type="file" multiple onChange={handleImageUpload} />

          <div className="image-grid">
            {images.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="training" />
                <button
                  className="delete-btn"
                  onClick={() => removeImage(index)}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT HALF – TRAINING + TESTING */}
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
              <p className="note">Training in progress...</p>
            )}

            {trainingStatus === "trained" && (
              <>
                <p className="success-text">Training Complete ✅</p>

                {/* 🔥 NEW BUTTON: Test Model */}
                <button
                  className="btn secondary"
                  onClick={() => navigate("/test")}
                >
                  Test Model
                </button>
              </>
            )}

            <p className="note">
              Minimum 25 normal images required for training
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingPage;
