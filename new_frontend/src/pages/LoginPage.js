import React from "react";
import "../styles/Auth.css";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { loginUser } from "../utils/auth";

const LoginPage = () => {
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = location.state?.from || "/train";

  const handleLogin = () => {
    if (!email.trim() || !password.trim()) {
      alert("Please enter email and password.");
      return;
    }

    fetch("http://localhost:5000/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: email.trim(),
        password: password.trim(),
      }),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.success) {
          throw new Error(data.message || "Login failed");
        }
        loginUser(data.user?.email || email.trim(), data.user?.name || "");
        navigate(redirectTo, { replace: true });
      })
      .catch((err) => {
        alert(err.message || "Login failed");
      });
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Login</h2>

        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button className="auth-btn" onClick={handleLogin}>Login</button>

        <p>
          New user?
          <Link to="/signup"> Create an account</Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
