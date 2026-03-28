import React, { useState, useEffect } from 'react';
import './ModelSelector.css';

const ModelSelector = ({ onModelChange }) => {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/');
      const data = await response.json();
      setModels(data.models || []);
      setCurrentModel(data.current_model || '');
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const loadModel = async (modelName) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/load-model/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });
      
      const data = await response.json();
      if (response.ok) {
        setCurrentModel(modelName);
        if (onModelChange) {
          onModelChange(modelName, data.threshold);
        }
        alert(`Model ${modelName} loaded successfully!`);
      } else {
        alert(`Error loading model: ${data.error}`);
      }
    } catch (error) {
      console.error('Error loading model:', error);
      alert('Failed to load model');
    }
    setLoading(false);
  };

  return (
    <div className="model-selector">
      <h3>Available Models</h3>
      {models.length === 0 ? (
        <p>No trained models found. Train a model first.</p>
      ) : (
        <ul className="model-list">
          {models.map((model) => (
            <li key={model.name} className="model-item">
              <div className="model-info">
                <strong>{model.name}</strong>
                <span className="model-date">
                  {new Date(model.created * 1000).toLocaleString()}
                </span>
                {model.threshold && (
                  <span className="model-threshold">
                    Threshold: {model.threshold.toFixed(4)}
                  </span>
                )}
              </div>
              <button
                onClick={() => loadModel(model.name)}
                disabled={loading || currentModel === model.name}
                className={currentModel === model.name ? 'active' : ''}
              >
                {currentModel === model.name ? 'Active' : 'Load'}
              </button>
            </li>
          ))}
        </ul>
      )}
      <button 
        onClick={fetchModels} 
        className="refresh-btn"
        disabled={loading}
      >
        Refresh List
      </button>
    </div>
  );
};

export default ModelSelector;