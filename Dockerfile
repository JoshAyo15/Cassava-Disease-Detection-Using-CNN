FROM python:3.10-slim

WORKDIR /app

# Install system libraries needed for ML
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]