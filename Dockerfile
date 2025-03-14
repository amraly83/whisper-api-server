# Stage 1: Build
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . /app

# Environment variables
ENV WHISPER_MODEL=${WHISPER_MODEL:-base}
ENV API_KEY=${API_KEY:?ERR_API_KEY_NOT_SET}
ENV PORT=${PORT:-8088}
ENV HOST=${HOST:-0.0.0.0}
ENV UPLOAD_DIR=${UPLOAD_DIR:-/app/uploads}
ENV DEBUG=${DEBUG:-false}

# Security settings
# Maximum file size (default: 50MB)
ENV MAX_FILE_SIZE=${MAX_FILE_SIZE:-52428800}

# Allowed CORS origins (default: *)
ENV ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}

# Rate limiting configuration (default: 10/minute,50/hour)
ENV RATE_LIMITS=${RATE_LIMITS:-"10/minute,50/hour"}

# Create upload directory
RUN mkdir -p ${UPLOAD_DIR}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8088/health || exit 1

# Optimize Uvicorn workers based on memory constraints
ENV UVICORN_WORKERS=2

# Start command with optimized worker configuration
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "$UVICORN_WORKERS"]
