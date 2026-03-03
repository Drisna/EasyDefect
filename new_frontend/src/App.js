import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import TrainingPage from "./pages/TrainingPage";
import TestingPage from "./pages/TestingPage";
import NotFound from "./pages/NotFound";

function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [backendError, setBackendError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/health/")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        console.log("Backend connected:", data);
        setBackendStatus(data);
      })
      .catch((err) => {
        console.error("Backend connection failed:", err);
        setBackendError(err.message);
      });
  }, []);

  return (
    <Router>
      <Navbar />

      {/* 🔥 FIX: Only show status bar if there's something to show,
          and use dark theme colours consistent with the rest of the UI */}
      {(backendStatus || backendError) && (
        <div style={{
          padding: "8px 20px",
          backgroundColor: "#0f172a",
          borderBottom: "1px solid #1f2937",
          textAlign: "center",
          fontSize: "13px",
          marginTop: "64px"   // clears the fixed navbar
        }}>
          {backendStatus && (
            <span style={{ color: "#22c55e" }}>
              ✅ Backend connected: {backendStatus.message}
            </span>
          )}
          {backendError && (
            <span style={{ color: "#ef4444" }}>
              ❌ Backend error: {backendError}
            </span>
          )}
        </div>
      )}

      <div className="main-body">
        <Routes>
          <Route path="/" element={<HomePage backendStatus={backendStatus} />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/train" element={<TrainingPage />} />
          <Route path="/test" element={<TestingPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>

      <Footer />
    </Router>
  );
}

export default App;