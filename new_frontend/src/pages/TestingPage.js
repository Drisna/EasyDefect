import React, { useState, useEffect } from 'react';
import ModelSelector from '../components/ModelSelector';
import './Testing.css'; // Make sure you have this CSS file

const Testing = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [testResults, setTestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewImages, setPreviewImages] = useState([]);

  const handleModelChange = (modelName, threshold) => {
    setSelectedModel({ name: modelName, threshold });
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    
    // Create preview URLs
    const previews = files.map(file => URL.createObjectURL(file));
    setPreviewImages(previews);
  };

  const runTest = async () => {
    if (!selectedModel) {
      alert('Please select a model first!');
      return;
    }

    if (selectedFiles.length === 0) {
      alert('Please select test images!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    
    // Add all selected files to formData
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
    
    try {
      const response = await fetch('http://localhost:5000/api/predict/', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setTestResults(data);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Test failed:', error);
      alert('Test failed. Make sure the backend server is running.');
    }
    setLoading(false);
  };

  // Cleanup preview URLs when component unmounts
  useEffect(() => {
    return () => {
      previewImages.forEach(url => URL.revokeObjectURL(url));
    };
  }, [previewImages]);

  return (
    <div className="testing-page">
      <h1>Model Testing</h1>
      
      <div className="model-selection-section">
        <h2>1. Select Model for Testing</h2>
        <ModelSelector onModelChange={handleModelChange} />
      </div>

      {selectedModel && (
        <div className="active-model-info">
          <p>Active Model: <strong>{selectedModel.name}</strong></p>
          <p>Threshold: <strong>{selectedModel.threshold?.toFixed(4)}</strong></p>
        </div>
      )}

      <div className="test-controls">
        <h2>2. Upload Test Images</h2>
        <input 
          type="file" 
          multiple 
          accept="image/*"
          onChange={handleFileSelect}
          disabled={!selectedModel}
        />
        
        {previewImages.length > 0 && (
          <div className="image-previews">
            <h3>Selected Images ({previewImages.length})</h3>
            <div className="preview-grid">
              {previewImages.map((url, index) => (
                <div key={index} className="preview-item">
                  <img src={url} alt={`preview-${index}`} />
                </div>
              ))}
            </div>
          </div>
        )}

        <button 
          onClick={runTest} 
          disabled={!selectedModel || selectedFiles.length === 0 || loading}
          className="test-button"
        >
          {loading ? 'Testing...' : 'Run Test'}
        </button>
      </div>

      {testResults && (
        <div className="test-results">
          <h2>3. Test Results</h2>
          <div className="accuracy-display">
            <div className="accuracy-card normal">
              <h3>Normal Products</h3>
              <p className="accuracy">{testResults.summary.normal_accuracy}</p>
              <p>Predicted Correctly</p>
              <p className="count">Count: {testResults.summary.normal}</p>
            </div>
            <div className="accuracy-card defective">
              <h3>Defective Products</h3>
              <p className="accuracy">{testResults.summary.defective_accuracy}</p>
              <p>Predicted Correctly</p>
              <p className="count">Count: {testResults.summary.defective}</p>
            </div>
          </div>
          
          <div className="total-accuracy">
            <h3>Overall Accuracy</h3>
            <p className="big-accuracy">
              {((testResults.summary.normal + testResults.summary.defective) / testResults.summary.total * 100).toFixed(2)}%
            </p>
            <p>Total Images: {testResults.summary.total}</p>
          </div>
          
          <p className="model-info">
            Using model: <strong>{testResults.model_used}</strong> (threshold: {testResults.threshold.toFixed(4)})
          </p>

          <div className="detailed-results">
            <h3>Detailed Predictions</h3>
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Prediction</th>
                  <th>Error Score</th>
                  <th>Threshold</th>
                </tr>
              </thead>
              <tbody>
                {testResults.results.map((result, index) => (
                  <tr key={index}>
                    <td>{result.filename}</td>
                    <td className={result.prediction}>
                      {result.prediction}
                    </td>
                    <td>{result.error.toFixed(4)}</td>
                    <td>{result.threshold.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Testing;