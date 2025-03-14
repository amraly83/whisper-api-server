# Use official Python slim image
FROM python:3.9-slim

# Install system dependencies including curl
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV WHISPER_MODEL=base
ENV API_KEY=please-set-in-docker-compose
ENV PORT=8088
ENV HOST=0.0.0.0
ENV UPLOAD_DIR=/app/uploads
ENV DEBUG=false

# Create upload directory
RUN mkdir -p ${UPLOAD_DIR}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8088/health || exit 1

# Start command
CMD uvicorn server:app --host 0.0.0.0 --port $PORT
