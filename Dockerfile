# Dockerfile for AI Surveillance Anomaly Detection System
FROM python:3.10-slim

# Install system dependencies for OpenCV and video codec acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI backend & UI port
EXPOSE 8000

ENV MODEL_PATH="best_model.keras"
ENV HF_REPO_ID="SantoshDN/ai-enable-anomaly-detection"

# Start FastAPI server
CMD ["python", "-m", "backend.main"]
