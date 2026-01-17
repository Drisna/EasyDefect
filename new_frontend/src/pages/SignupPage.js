import React from "react";
import "../styles/Auth.css";
import { Link } from "react-router-dom";

const SignupPage = () => {
  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Create Account</h2>

        <input type="text" placeholder="Full Name" />
        <input type="email" placeholder="Email" />
        <input type="password" placeholder="Password" />

        <button className="auth-btn">Signup</button>

        <p>
          Already registered?
          <Link to="/login"> Login here</Link>
        </p>
      </div>
    </div>
  );
};

export default SignupPage;
