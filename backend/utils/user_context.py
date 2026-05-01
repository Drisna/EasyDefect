import re
from flask import request


def get_request_user_email():
    email = (request.headers.get("X-User-Email") or "").strip().lower()
    return email


def get_user_storage_key(email: str):
    if not email:
        return ""
    # Keep filesystem-safe key for per-user folders.
    return re.sub(r"[^a-z0-9_.-]", "_", email)
