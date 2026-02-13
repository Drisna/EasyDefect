// import React, { useState } from "react";
// import "../styles/TestingPage.css";

// const TestingPage = () => {
//   const [normalImages, setNormalImages] = useState([]);
//   const [defectiveImages, setDefectiveImages] = useState([]);

//   const [accuracy, setAccuracy] = useState(null);
//   const [normalCorrect, setNormalCorrect] = useState(0);
//   const [defectiveCorrect, setDefectiveCorrect] = useState(0);

//   const [modelName, setModelName] = useState("EasyDefect_Model");

//   const handleNormalUpload = (e) => {
//     const files = Array.from(e.target.files);
//     const newImages = files.map((file) => ({
//       file,
//       url: URL.createObjectURL(file),
//       prediction: "Not tested",
//     }));
//     setNormalImages((prev) => [...prev, ...newImages]);
//   };

//   const handleDefectiveUpload = (e) => {
//     const files = Array.from(e.target.files);
//     const newImages = files.map((file) => ({
//       file,
//       url: URL.createObjectURL(file),
//       prediction: "Not tested",
//     }));
//     setDefectiveImages((prev) => [...prev, ...newImages]);
//   };

//   const removeNormal = (index) => {
//     setNormalImages(normalImages.filter((_, i) => i !== index));
//   };

//   const removeDefective = (index) => {
//     setDefectiveImages(defectiveImages.filter((_, i) => i !== index));
//   };

//   const handleTest = () => {
//     const total = normalImages.length + defectiveImages.length;

//     if (total < 30) {
//       alert("Please upload at least 30 images to test.");
//       return;
//     }

//     // Simulated prediction results
//     const updatedNormal = normalImages.map((img) => ({
//       ...img,
//       prediction: "Normal",
//     }));

//     const updatedDefective = defectiveImages.map((img) => ({
//       ...img,
//       prediction: "Defective",
//     }));

//     setNormalImages(updatedNormal);
//     setDefectiveImages(updatedDefective);

//     const nCorrect = updatedNormal.length;
//     const dCorrect = updatedDefective.length;

//     setNormalCorrect(nCorrect);
//     setDefectiveCorrect(dCorrect);

//     const acc = ((nCorrect + dCorrect) / total) * 100;
//     setAccuracy(acc.toFixed(2));
//   };

//   const handleDownload = () => {
//     const content = `
//       Model Name: ${modelName}
//       Accuracy: ${accuracy ? accuracy + "%" : "Not tested"}
//       Normal Correct: ${normalCorrect} / ${normalImages.length}
//       Defective Correct: ${defectiveCorrect} / ${defectiveImages.length}
//     `;

//     const blob = new Blob([content], { type: "text/plain" });
//     const link = document.createElement("a");
//     link.href = URL.createObjectURL(blob);
//     link.download = `${modelName}_result.txt`;
//     link.click();
//   };

//   return (
//     <div className="testing-page">
//       <h1>Test Your Model</h1>

//       <div className="testing-container">
//         {/* LEFT HALF */}
//         <div className="test-image-panel">
//           <h3>Normal Images</h3>
//           <input type="file" multiple onChange={handleNormalUpload} />

//           <div className="image-grid">
//             {normalImages.map((img, index) => (
//               <div key={index} className="image-card">
//                 <img src={img.url} alt="normal" />
//                 <button className="delete-btn" onClick={() => removeNormal(index)}>
//                   ✕
//                 </button>
//                 <p className="prediction normal">{img.prediction}</p>
//               </div>
//             ))}
//           </div>

//           <h3 style={{ marginTop: "25px" }}>Defective Images</h3>
//           <input type="file" multiple onChange={handleDefectiveUpload} />

//           <div className="image-grid">
//             {defectiveImages.map((img, index) => (
//               <div key={index} className="image-card">
//                 <img src={img.url} alt="defective" />
//                 <button className="delete-btn" onClick={() => removeDefective(index)}>
//                   ✕
//                 </button>
//                 <p className="prediction defective">{img.prediction}</p>
//               </div>
//             ))}
//           </div>
//         </div>

//         {/* RIGHT HALF */}
//         <div className="test-panel">
//           <div className="test-center">
//             <h3>Run Test</h3>

//             <button className="btn primary" onClick={handleTest}>
//               Test Model
//             </button>

//             <div className="result-box">
//               <p className="accuracy-text">
//                 Accuracy: {accuracy ? `${accuracy}%` : "Not tested yet"}
//               </p>

//               <p className="report-text">
//                 Normal Predicted Correctly: {normalCorrect} / {normalImages.length}
//               </p>
//               <p className="report-text">
//                 Defective Predicted Correctly: {defectiveCorrect} / {defectiveImages.length}
//               </p>
//             </div>

//             {accuracy && (
//               <button className="btn secondary" onClick={handleDownload}>
//                 Download Model
//               </button>
//             )}

//             <p className="note">
//               Minimum 30 images required for testing
//             </p>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default TestingPage;


import React, { useState, useEffect } from "react";
import "../styles/TestingPage.css";

const TestingPage = () => {
  const [normalImages, setNormalImages] = useState([]);
  const [defectiveImages, setDefectiveImages] = useState([]);

  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");

  const [accuracy, setAccuracy] = useState(null);
  const [normalCorrect, setNormalCorrect] = useState(0);
  const [defectiveCorrect, setDefectiveCorrect] = useState(0);

  // 🔥 Fetch trained models from backend
  useEffect(() => {
    fetch("http://localhost:5000/api/models/")
      .then(res => res.json())
      .then(data => {
        setModels(data.models);
        if (data.models.length > 0) {
          setModelName(data.models[0]);
        }
      });
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

  const handleTest = async () => {
    if (!modelName) {
      alert("Please select a model");
      return;
    }

    const formData = new FormData();
    formData.append("model_name", modelName);

    normalImages.forEach(img => {
      formData.append("normal_files", img.file);
    });

    defectiveImages.forEach(img => {
      formData.append("defective_files", img.file);
    });

    const response = await fetch("http://localhost:5000/api/test/", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    setAccuracy(data.accuracy);

    const updatedNormal = normalImages.map(img => {
      const result = data.results.find(r => r.filename === img.file.name);
      return { ...img, prediction: result?.prediction || "Error" };
    });

    const updatedDefective = defectiveImages.map(img => {
      const result = data.results.find(r => r.filename === img.file.name);
      return { ...img, prediction: result?.prediction || "Error" };
    });

    setNormalImages(updatedNormal);
    setDefectiveImages(updatedDefective);

    const nCorrect = data.results.filter(r => r.actual === "Normal" && r.prediction === "Normal").length;
    const dCorrect = data.results.filter(r => r.actual === "Defective" && r.prediction === "Defective").length;

    setNormalCorrect(nCorrect);
    setDefectiveCorrect(dCorrect);
  };

  return (
    <div className="testing-page">
      <h1>Test Your Model</h1>

      {/* 🔥 Model Selector */}
      <div style={{ marginBottom: "20px" }}>
        <label>Select Model: </label>
        <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
          {models.map((model, index) => (
            <option key={index} value={model}>{model}</option>
          ))}
        </select>
      </div>

      <div className="testing-container">
        <div className="test-image-panel">
          <h3>Normal Images</h3>
          <input type="file" multiple onChange={handleNormalUpload} />

          <div className="image-grid">
            {normalImages.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="normal" />
                <p className="prediction">{img.prediction}</p>
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: "25px" }}>Defective Images</h3>
          <input type="file" multiple onChange={handleDefectiveUpload} />

          <div className="image-grid">
            {defectiveImages.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.url} alt="defective" />
                <p className="prediction">{img.prediction}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="test-panel">
          <button className="btn primary" onClick={handleTest}>
            Test Model
          </button>

          <div className="result-box">
            <p>Accuracy: {accuracy ? `${accuracy}%` : "Not tested yet"}</p>
            <p>Normal Correct: {normalCorrect}</p>
            <p>Defective Correct: {defectiveCorrect}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestingPage;
