FROM python:3.10-slim

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8 nano model so container is self-contained offline
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application source
COPY . .

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

# Launch deployment server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
