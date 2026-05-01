import json
import os
from flask import Blueprint, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

auth_bp = Blueprint("auth", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _ensure_users_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def _load_users():
    _ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []


def _save_users(users):
    _ensure_users_file()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def get_registered_user_display_name(email):
    """Return stored full name for an email (used by model bundle download)."""
    email_norm = (email or "").strip().lower()
    if not email_norm:
        return ""
    for user in _load_users():
        if user.get("email") == email_norm:
            return (user.get("name") or "").strip()
    return ""


@auth_bp.route("/signup", methods=["POST"])
def signup():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not name or not email or not password:
        return jsonify({"success": False, "message": "name, email and password are required"}), 400

    users = _load_users()
    if any(user.get("email") == email for user in users):
        return jsonify({"success": False, "message": "Email is already registered"}), 409

    users.append({
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
    })
    _save_users(users)

    return jsonify({
        "success": True,
        "message": "Signup successful",
        "user": {"name": name, "email": email},
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "message": "email and password are required"}), 400

    users = _load_users()
    user = next((u for u in users if u.get("email") == email), None)
    if not user:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    if not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {"name": user.get("name"), "email": user.get("email")},
    })
