import React from "react";
import "../styles/HomePage.css";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <div className="home">
      {/* HERO SECTION */}
      <section className="hero">
        <h1>Detect Defects Effortlessly</h1>
        <p>
          EasyDefect helps you train AI models and detect anomalies in images
          quickly and accurately.
        </p>

        <div className="home-buttons">
          <div className="home-btn-card">
            <Link className="btn primary" to="/train">
              Train a New Model
            </Link>
          </div>

          <div className="home-btn-card">
            <Link className="btn secondary" to="/test">
              Test Trained Model
            </Link>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="steps">
        <h2>How EasyDefect Works</h2>

        <div className="steps-grid">
          <div className="card">
            <h3>1️⃣ Upload Training Images</h3>
            <p>
              Select at least 25 normal images to prepare your training dataset.
            </p>
          </div>

          <div className="card">
            <h3>2️⃣ Train the Model</h3>
            <p>
              Our system trains an anomaly detection model based on your data.
            </p>
          </div>

          <div className="card">
            <h3>3️⃣ Test & Analyze</h3>
            <p>
              Upload test images, view accuracy, and download the trained model.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
