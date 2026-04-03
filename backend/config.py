"""
Backend configuration using .env file.
Load with: from dotenv import load_dotenv
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/cassava_model.keras")
INDICES_PATH = os.getenv("INDICES_PATH", "models/class_indices.json")

# CORS settings – restrict to frontend in production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Image upload
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10_485_760))  # 10 MB

# Server
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Validate required files exist
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run 'python model/train.py' first, then copy models/ to backend/models/"
    )
