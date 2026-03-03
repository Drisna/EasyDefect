import React, { useState, useEffect } from "react";
import "../styles/TestingPage.css";

const API = "http://localhost:5000";

const TestingPage = () => {
  const [normalImages,    setNormalImages]    = useState([]);
  const [defectiveImages, setDefectiveImages] = useState([]);
  const [models,    setModels]    = useState([]);
  const [modelName, setModelName] = useState("");
  const [accuracy,  setAccuracy]  = useState(null);
  const [normalCorrect,    setNormalCorrect]    = useState(0);
  const [defectiveCorrect, setDefectiveCorrect] = useState(0);
  const [totalNormal,    setTotalNormal]    = useState(0);
  const [totalDefective, setTotalDefective] = useState(0);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  // Fetch available trained models on mount
  useEffect(() => {
    fetch(`${API}/api/models/`)
      .then(res => res.json())
      .then(data => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setModelName(data.models[0]);
        }
      })
      .catch(err => console.error("Could not fetch models:", err));
  }, []);

  const handleNormalUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "Not tested",
    }));
    setNormalImages(prev => [...prev, ...newImages]);
  };

  const handleDefectiveUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "Not tested",
    }));
    setDefectiveImages(prev => [...prev, ...newImages]);
  };

  const removeNormal    = (i) => setNormalImages(normalImages.filter((_, idx) => idx !== i));
  const removeDefective = (i) => setDefectiveImages(defectiveImages.filter((_, idx) => idx !== i));

  const handleTest = async () => {
    if (!modelName) {
      alert("Please select a trained model first.");
      return;
    }

    // 🔥 FIX: validate at least 1 image before calling API
    if (normalImages.length === 0 && defectiveImages.length === 0) {
      alert("Please upload at least one image to test.");
      return;
    }

    setLoading(true);
    setAccuracy(null);

    const formData = new FormData();
    formData.append("model_name", modelName);
    normalImages.forEach(img    => formData.append("normal_files",    img.file));
    defectiveImages.forEach(img => formData.append("defective_files", img.file));

    try {
      const response = await fetch(`${API}/api/test/`, { method: "POST", body: formData });
      const data = await response.json();

      if (data.error) {
        alert(`Test failed: ${data.error}`);
        setLoading(false);
        return;
      }

      setResults(data.results);
      setAccuracy(data.accuracy);

      // Update per-image predictions in state
      setNormalImages(prev =>
        prev.map(img => {
          const r = data.results.find(r => r.filename === img.file.name);
          return { ...img, prediction: r ? r.prediction : "Error" };
        })
      );
      setDefectiveImages(prev =>
        prev.map(img => {
          const r = data.results.find(r => r.filename === img.file.name);
          return { ...img, prediction: r ? r.prediction : "Error" };
        })
      );

      const nCorrect = data.results.filter(r => r.actual === "Normal"    && r.correct).length;
      const dCorrect = data.results.filter(r => r.actual === "Defective" && r.correct).length;
      setNormalCorrect(nCorrect);
      setDefectiveCorrect(dCorrect);
      setTotalNormal(normalImages.length);
      setTotalDefective(defectiveImages.length);

    } catch (err) {
      console.error("Test error:", err);
      alert("Testing failed. Is the backend running?");
    }

    setLoading(false);
  };

  // 🔥 FIX: Download model as zip from backend
  const handleDownload = () => {
    if (!modelName) return;
    window.open(`${API}/api/models/download/${modelName}`, "_blank");
  };

  return (
    <div className="testing-page">
      <h1>Test Your Model</h1>

      {/* 🔥 FIX: Styled model selector consistent with dark theme */}
      <div className="model-selector">
        <label>Select Model:</label>
        {models.length === 0 ? (
          <span className="no-models">No trained models found. Train a model first.</span>
        ) : (
          <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
            {models.map((model, i) => (
              <option key={i} value={model}>{model}</option>
            ))}
          </select>
        )}
      </div>

      <div className="testing-container">

        {/* LEFT — Image Panels */}
        <div className="test-image-panel">
          <h3>Normal Images</h3>
          <p className="count-text">{normalImages.length} selected</p>
          <input type="file" multiple accept=".jpg,.jpeg,.png,.bmp" onChange={handleNormalUpload} />

          <div className="image-grid">
            {normalImages.map((img, i) => (
              <div key={i} className="image-card">
                <img src={img.url} alt="normal" />
                <button className="delete-btn" onClick={() => removeNormal(i)}>✕</button>
                <p className={`prediction ${
                  img.prediction === "Normal" ? "pred-normal" :
                  img.prediction === "Defective" ? "pred-defective" : ""
                }`}>
                  {img.prediction}
                </p>
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: "25px" }}>Defective Images</h3>
          <p className="count-text">{defectiveImages.length} selected</p>
          <input type="file" multiple accept=".jpg,.jpeg,.png,.bmp" onChange={handleDefectiveUpload} />

          <div className="image-grid">
            {defectiveImages.map((img, i) => (
              <div key={i} className="image-card">
                <img src={img.url} alt="defective" />
                <button className="delete-btn" onClick={() => removeDefective(i)}>✕</button>
                <p className={`prediction ${
                  img.prediction === "Normal" ? "pred-normal" :
                  img.prediction === "Defective" ? "pred-defective" : ""
                }`}>
                  {img.prediction}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT — Controls & Results */}
        <div className="test-panel">
          <div className="test-center">
            <h3>Run Test</h3>

            <button
              className="btn primary"
              onClick={handleTest}
              disabled={loading || models.length === 0}
            >
              {loading ? "Testing…" : "Test Model"}
            </button>

            {accuracy !== null && (
              <div className="result-box">
                <p className="accuracy-text">Accuracy: {accuracy}%</p>
                <p className="report-text">
                  Normal correct: {normalCorrect} / {totalNormal}
                </p>
                <p className="report-text">
                  Defective correct: {defectiveCorrect} / {totalDefective}
                </p>

              
                {/* 🔥 FIX: Download model button — was missing from rewrite */}
                <button className="btn secondary download-btn" onClick={handleDownload}>
                  ⬇ Download Model
                </button>
              </div>
            )}

            <p className="note">Upload normal and/or defective images, then click Test Model.</p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default TestingPage;