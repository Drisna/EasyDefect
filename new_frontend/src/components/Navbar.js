// src/components/Navbar.js
import React from "react";
import "../styles/Navbar.css";
import { Link, useNavigate } from "react-router-dom";
import { isAuthenticated, logoutUser } from "../utils/auth";

const Navbar = () => {
  const navigate = useNavigate();
  const loggedIn = isAuthenticated();

  const handleLogout = () => {
    logoutUser();
    navigate("/login");
  };

  return (
    <header className="nav-container">
      <div className="logo">
        <span role="img" aria-label="robot" className="robot-icon">🤖</span>
        EasyDefect
      </div>

      <input type="checkbox" id="menu-toggle" />
      <label className="menu-icon" htmlFor="menu-toggle">&#9776;</label>

      <ul className="nav-links">
        <li><Link to="/">Home</Link></li>
        <li><Link to={loggedIn ? "/train" : "/login"}>Training</Link></li>
        <li><Link to={loggedIn ? "/test" : "/login"}>Testing</Link></li>
        {loggedIn ? (
          <li>
            <button type="button" className="login-btn" onClick={handleLogout}>
              Logout
            </button>
          </li>
        ) : (
          <li><Link to="/login" className="login-btn">Login</Link></li>
        )}
      </ul>
    </header>
  );
};

export default Navbar;
