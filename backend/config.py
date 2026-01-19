import os
from dotenv import load_dotenv

# Load .env file (works on Windows too)
load_dotenv()


class Config:
    # Flask settings
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
    TESTING = os.getenv("FLASK_TESTING", "0") == "1"

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "easydefect-secret-key")

    # Database (optional)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
