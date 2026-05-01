import React, { useState, useEffect } from "react";
import "../styles/TestingPage.css";
import { getAuthHeaders, getCurrentUser, getCurrentUserDisplayName } from "../utils/auth";

const TestingPage = () => {
  const [normalImages,    setNormalImages]    = useState([]);
  const [defectiveImages, setDefectiveImages] = useState([]);

  const [models,    setModels]    = useState([]);
  const [modelName, setModelName] = useState("");

  const [accuracy,        setAccuracy]        = useState(null);
  const [normalCorrect,   setNormalCorrect]   = useState(0);
  const [defectiveCorrect,setDefectiveCorrect]= useState(0);
  const [totalTested,     setTotalTested]     = useState(0);

  const [isTesting,  setIsTesting]  = useState(false);
  const [errorMsg,   setErrorMsg]   = useState("");
  const [testDone,   setTestDone]   = useState(false);

  // ── Fetch trained model list from backend ────────────────────────────────
  useEffect(() => {
    fetch("http://localhost:5000/api/models/", { headers: getAuthHeaders() })
      .then(res => res.json())
      .then(data => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setModelName(data.models[0]);
        }
      })
      .catch(() => setErrorMsg("Could not load models. Is the backend running?"));
  }, []);

  const handleNormalUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map(file => ({
      file,
      url:        URL.createObjectURL(file),
      prediction: "Not tested",
      correct:    null,
    }));
    setNormalImages(prev => [...prev, ...newImages]);
    setTestDone(false);
  };

  const handleDefectiveUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map(file => ({
      file,
      url:        URL.createObjectURL(file),
      prediction: "Not tested",
      correct:    null,
    }));
    setDefectiveImages(prev => [...prev, ...newImages]);
    setTestDone(false);
  };

  const removeNormal    = (i) => setNormalImages(normalImages.filter((_, idx) => idx !== i));
  const removeDefective = (i) => setDefectiveImages(defectiveImages.filter((_, idx) => idx !== i));

  // ── Run test ─────────────────────────────────────────────────────────────
  const handleTest = async () => {
    setErrorMsg("");

    if (!modelName) {
      setErrorMsg("Please select a model first.");
      return;
    }
    if (normalImages.length === 0 && defectiveImages.length === 0) {
      setErrorMsg("Upload at least one image to test.");
      return;
    }

    setIsTesting(true);

    const formData = new FormData();
    formData.append("model_name", modelName);
    normalImages.forEach(img    => formData.append("normal_files",    img.file));
    defectiveImages.forEach(img => formData.append("defective_files", img.file));

    try {
      const response = await fetch("http://localhost:5000/api/test/", {
        method: "POST",
        headers: getAuthHeaders(),
        body:   formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setErrorMsg(data.error || "Testing failed.");
        setIsTesting(false);
        return;
      }

      // ── Map results back to images ──────────────────────────────────────
      // Backend uses "Normal" / "Defective" — match exactly
      const updatedNormal = normalImages.map(img => {
        const result = data.results.find(
          r => r.filename === img.file.name && r.actual === "Normal"
        );
        return {
          ...img,
          prediction: result?.prediction ?? "Error",
          correct:    result?.correct    ?? false,
        };
      });

      const updatedDefective = defectiveImages.map(img => {
        const result = data.results.find(
          r => r.filename === img.file.name && r.actual === "Defective"
        );
        return {
          ...img,
          prediction: result?.prediction ?? "Error",
          correct:    result?.correct    ?? false,
        };
      });

      setNormalImages(updatedNormal);
      setDefectiveImages(updatedDefective);

      setAccuracy(data.accuracy);
      setTotalTested(data.total);

      const nCorrect = data.results.filter(r => r.actual === "Normal"    && r.correct).length;
      const dCorrect = data.results.filter(r => r.actual === "Defective" && r.correct).length;
      setNormalCorrect(nCorrect);
      setDefectiveCorrect(dCorrect);

      setTestDone(true);

    } catch (err) {
      setErrorMsg("Network error. Make sure the backend is running.");
      console.error(err);
    } finally {
      setIsTesting(false);
    }
  };

  // ── Download model zip from backend ──────────────────────────────────────
  const handleDownload = () => {
    if (!modelName) return;
    const email = encodeURIComponent(getCurrentUser() || "");
    const displayName = encodeURIComponent(getCurrentUserDisplayName() || "");
    window.location.href = `http://localhost:5000/api/models/download/${modelName}?user_email=${email}&display_name=${displayName}`;
  };

  // ── Label colour helper ───────────────────────────────────────────────────
  const labelClass = (prediction) => {
    if (prediction === "Normal")    return "prediction normal-label";
    if (prediction === "Defective") return "prediction defective-label";
    return "prediction";
  };

  return (
    <div className="testing-page">
      <h1>Test Your Model</h1>

      {/* Model selector */}
      <div className="model-selector">
        <label>Select Model: </label>
        {models.length === 0 ? (
          <span className="no-models">No trained models found. Train a model first.</span>
        ) : (
          <select value={modelName} onChange={e => setModelName(e.target.value)}>
            {models.map((m, i) => (
              <option key={i} value={m}>{m}</option>
            ))}
          </select>
        )}
      </div>

      {errorMsg && <p className="error-text">{errorMsg}</p>}

      <div className="testing-container">

        {/* ── LEFT: Image panels ─────────────────────────────────────────── */}
        <div className="test-image-panel">

          <h3>Normal Images ({normalImages.length})</h3>
          <input
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.bmp"
            onChange={handleNormalUpload}
            disabled={isTesting}
          />
          <div className="image-grid">
            {normalImages.map((img, i) => (
              <div key={i} className={`image-card ${img.correct === false && testDone ? "wrong" : img.correct ? "right" : ""}`}>
                <img src={img.url} alt="normal" />
                <button className="delete-btn" onClick={() => removeNormal(i)} disabled={isTesting}>✕</button>
                <p className={labelClass(img.prediction)}>{img.prediction}</p>
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: "25px" }}>Defective Images ({defectiveImages.length})</h3>
          <input
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.bmp"
            onChange={handleDefectiveUpload}
            disabled={isTesting}
          />
          <div className="image-grid">
            {defectiveImages.map((img, i) => (
              <div key={i} className={`image-card ${img.correct === false && testDone ? "wrong" : img.correct ? "right" : ""}`}>
                <img src={img.url} alt="defective" />
                <button className="delete-btn" onClick={() => removeDefective(i)} disabled={isTesting}>✕</button>
                <p className={labelClass(img.prediction)}>{img.prediction}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── RIGHT: Controls + results ──────────────────────────────────── */}
        <div className="test-panel">
          <div className="test-center">
            <h3>Run Test</h3>

            <button
              className="btn primary"
              onClick={handleTest}
              disabled={isTesting || models.length === 0}
            >
              {isTesting ? "Testing…" : "Test Model"}
            </button>

            <div className="result-box">
              <p className="accuracy-text">
                Accuracy: {accuracy !== null ? `${accuracy}%` : "Not tested yet"}
              </p>
              <p className="report-text">
                Normal Correct: {normalCorrect} / {normalImages.length}
              </p>
              <p className="report-text">
                Defective Correct: {defectiveCorrect} / {defectiveImages.length}
              </p>
              {totalTested > 0 && (
                <p className="report-text">Total Tested: {totalTested}</p>
              )}
            </div>

            {/* Download model button — only shown after testing */}
            {testDone && (
              <button className="btn secondary" onClick={handleDownload}>
                ⬇ Download Model
              </button>
            )}

            <p className="note">
              Upload normal and/or defective images, then click "Test Model"
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default TestingPage;
