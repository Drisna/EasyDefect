import React from "react";
import "../styles/Auth.css";
import { Link } from "react-router-dom";

const LoginPage = () => {
  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Login</h2>

        <input type="email" placeholder="Email" />
        <input type="password" placeholder="Password" />

        <button className="auth-btn">Login</button>

        <p>
          New user?
          <Link to="/signup"> Create an account</Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
