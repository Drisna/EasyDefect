import React from "react";
import "../styles/Auth.css";
import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../utils/auth";

const SignupPage = () => {
  const [fullName, setFullName] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const navigate = useNavigate();

  const handleSignup = () => {
    if (!fullName.trim() || !email.trim() || !password.trim()) {
      alert("Please fill in all fields.");
      return;
    }

    fetch("http://localhost:5000/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: fullName.trim(),
        email: email.trim(),
        password: password.trim(),
      }),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.success) {
          throw new Error(data.message || "Signup failed");
        }
        loginUser(data.user?.email || email.trim(), data.user?.name || "");
        navigate("/train", { replace: true });
      })
      .catch((err) => {
        alert(err.message || "Signup failed");
      });
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Create Account</h2>

        <input
          type="text"
          placeholder="Full Name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
        />
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

        <button className="auth-btn" onClick={handleSignup}>Signup</button>

        <p>
          Already registered?
          <Link to="/login"> Login here</Link>
        </p>
      </div>
    </div>
  );
};

export default SignupPage;
