// src/components/Navbar.js
import React from "react";
import "../styles/Navbar.css";
import { Link } from "react-router-dom";

const Navbar = () => {
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
        <li><Link to="/train">Training</Link></li>
        <li><Link to="/test">Testing</Link></li>
        <li><Link to="/login" className="login-btn">Login</Link></li>
      </ul>
    </header>
  );
};

export default Navbar;
