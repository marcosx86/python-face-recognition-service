"""
Configuration settings for the Facial Recognition Service.
Loads database connection parameters from environment variables
with development-friendly fallbacks.
"""
import os

class Config:
    """Base configuration class containing application settings."""
    # Database connection parameters set via environment variables.
    # The default values provided act as fallbacks if the env vars are missing.
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "somestrongpassword")
    DB_USER = os.environ.get("DB_USER", "facialrecognition")
    DB_NAME = os.environ.get("DB_NAME", "facialrecognition")
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    
    # Connection string for SQLAlchemy
    SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Storage directory for original face images
    FACE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_storage")

