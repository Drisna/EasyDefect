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
  // State to store backend response
  const [backendStatus, setBackendStatus] = useState(null);
  const [backendError, setBackendError] = useState(null);

  useEffect(() => {
    // Call the backend health API
    fetch("http://localhost:5000/api/health/")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
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

      {/* Display backend status at top of app */}
      <div style={{ padding: "10px", backgroundColor: "#f2f2f2" }}>
        {backendStatus && (
          <p style={{ color: "green" }}>
            Backend connected: {backendStatus.message}
          </p>
        )}
        {backendError && (
          <p style={{ color: "red" }}>Backend error: {backendError}</p>
        )}
      </div>

      <Routes>
        <Route path="/" element={<HomePage backendStatus={backendStatus} />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/train" element={<TrainingPage />} />
        <Route path="/test" element={<TestingPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
