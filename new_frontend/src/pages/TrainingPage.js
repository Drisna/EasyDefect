import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TrainingPage.css";

const TrainingPage = () => {
  const [images, setImages] = useState([]);
  const [modelName, setModelName] = useState("");
  const [trainingStatus, setTrainingStatus] = useState("idle"); // idle | training | trained
  const [uploadStatus, setUploadStatus] = useState({}); // track individual upload status

  const navigate = useNavigate();

  // Handle file selection and upload
  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      uploaded: false,
      message: "Uploading...",
    }));

    setImages((prev) => [...prev, ...newImages]);

    // Upload each file to backend
    for (const img of newImages) {
      const formData = new FormData();
      formData.append("file", img.file);

      try {
        const res = await fetch("http://localhost:5000/api/predict/", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();

        setImages((prev) =>
          prev.map((item) =>
            item === img
              ? { ...item, uploaded: data.success, message: data.message }
              : item
          )
        );
      } catch (err) {
        setImages((prev) =>
          prev.map((item) =>
            item === img
              ? { ...item, uploaded: false, message: "Upload failed" }
              : item
          )
        );
        console.error("Upload error:", err);
      }
    }
  };

  // Remove image from preview
  const removeImage = (index) => {
    setImages(images.filter((_, i) => i !== index));
  };

  // Handle training
  const handleTrain = async () => {
    if (!modelName.trim()) {
      alert("Please enter a model name");
      return;
    }

    if (images.length < 25) {
      alert("Minimum 25 images are required to train the model");
      return;
    }

    // Ensure all images are uploaded
    const notUploaded = images.filter((img) => !img.uploaded);
    if (notUploaded.length > 0) {
      alert("Please wait for all images to finish uploading before training");
      return;
    }

    setTrainingStatus("training");

    try {
      // Call backend to start training
      const response = await fetch("http://localhost:5000/api/train/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName, epochs: 50 }), // epochs can be dynamic
      });

      const data = await response.json();
      console.log(data);

      if (data.success) {
        setTrainingStatus("trained");
        alert(`Training Complete: ${data.message}`);
      } else {
        setTrainingStatus("idle");
        alert(`Training Failed: ${data.message}`);
      }
    } catch (err) {
      console.error("Error training model:", err);
      setTrainingStatus("idle");
      alert("Training failed due to server error");
    }
  };

  return (
    <div className="training-page">
      <h1>Train Your Model</h1>

      <div className="training-container">
        {/* LEFT HALF – IMAGE PREVIEW */}
        <div className="image-panel">
          <h3>Training Images</h3>
          <p className="count-text">{images.length} / 25 images selected</p>

          <input
            type="file"
            multiple
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
                <p
                  className={`upload-status ${
                    img.uploaded ? "success-text" : "error-text"
                  }`}
                >
                  {img.message}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT HALF – TRAINING SETUP */}
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
              <p className="note">Training in progress... This may take a few minutes</p>
            )}

            {trainingStatus === "trained" && (
              <>
                <p className="success-text">Training Complete ✅</p>
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
