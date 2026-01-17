import React, { useState } from "react";
import "../styles/TestingPage.css";

const TestingPage = () => {
  const [normalImages, setNormalImages] = useState([]);
  const [defectiveImages, setDefectiveImages] = useState([]);

  const [accuracy, setAccuracy] = useState(null);
  const [normalCorrect, setNormalCorrect] = useState(0);
  const [defectiveCorrect, setDefectiveCorrect] = useState(0);

  const [modelName, setModelName] = useState("EasyDefect_Model");

  const handleNormalUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "Not tested",
    }));
    setNormalImages((prev) => [...prev, ...newImages]);
  };

  const handleDefectiveUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "Not tested",
    }));
    setDefectiveImages((prev) => [...prev, ...newImages]);
  };

  const removeNormal = (index) => {
    setNormalImages(normalImages.filter((_, i) => i !== index));
  };

  const removeDefective = (index) => {
    setDefectiveImages(defectiveImages.filter((_, i) => i !== index));
  };

  const handleTest = () => {
    const total = normalImages.length + defectiveImages.length;

    if (total < 30) {
      alert("Please upload at least 30 images to test.");
      return;
    }

    // Simulated prediction results
    const updatedNormal = normalImages.map((img) => ({
      ...img,
      prediction: "Normal",
    }));

    const updatedDefective = defectiveImages.map((img) => ({
      ...img,
      prediction: "Defective",
    }));

    setNormalImages(updatedNormal);
    setDefectiveImages(updatedDefective);

    const nCorrect = updatedNormal.length;
    const dCorrect = updatedDefective.length;

    setNormalCorrect(nCorrect);
    setDefectiveCorrect(dCorrect);

    const acc = ((nCorrect + dCorrect) / total) * 100;
    setAccuracy(acc.toFixed(2));
  };

  const handleDownload = () => {
    const content = `
      Model Name: ${modelName}
      Accuracy: ${accuracy ? accuracy + "%" : "Not tested"}
      Normal Correct: ${normalCorrect} / ${normalImages.length}
      Defective Correct: ${defectiveCorrect} / ${defectiveImages.length}
    `;

    const blob = new Blob([content], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${modelName}_result.txt`;
    link.click();
  };

  return (
    <div className="testing-page">
      <h1>Test Your Model</h1>

      <div className="testing-container">
        {/* LEFT HALF */}
        <div className="test-image-panel">
          <h3>Normal Images</h3>
          <input type="file" multiple onChange={handleNormalUpload} />

          <div className="image-grid">
            {normalImages.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="normal" />
                <button className="delete-btn" onClick={() => removeNormal(index)}>
                  ✕
                </button>
                <p className="prediction normal">{img.prediction}</p>
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: "25px" }}>Defective Images</h3>
          <input type="file" multiple onChange={handleDefectiveUpload} />

          <div className="image-grid">
            {defectiveImages.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="defective" />
                <button className="delete-btn" onClick={() => removeDefective(index)}>
                  ✕
                </button>
                <p className="prediction defective">{img.prediction}</p>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT HALF */}
        <div className="test-panel">
          <div className="test-center">
            <h3>Run Test</h3>

            <button className="btn primary" onClick={handleTest}>
              Test Model
            </button>

            <div className="result-box">
              <p className="accuracy-text">
                Accuracy: {accuracy ? `${accuracy}%` : "Not tested yet"}
              </p>

              <p className="report-text">
                Normal Predicted Correctly: {normalCorrect} / {normalImages.length}
              </p>
              <p className="report-text">
                Defective Predicted Correctly: {defectiveCorrect} / {defectiveImages.length}
              </p>
            </div>

            {accuracy && (
              <button className="btn secondary" onClick={handleDownload}>
                Download Model
              </button>
            )}

            <p className="note">
              Minimum 30 images required for testing
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestingPage;
