# Stage 1: Build
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Stage 2: Final
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/* /usr/local/lib/python3.9/site-packages/
COPY . /app

# Environment variables (will be set via .env or docker-compose)
# Consider using a .env file or docker-compose.yml instead of hardcoding these
ENV WHISPER_MODEL=base
ENV API_KEY=default_api_key
ENV PORT=8088
ENV HOST=0.0.0.0
ENV UPLOAD_DIR=/app/uploads
ENV DEBUG=false
ENV MAX_FILE_SIZE=52428800
ENV ALLOWED_ORIGINS=*
ENV RATE_LIMITS="10/minute,50/hour"

# Create upload directory
RUN mkdir -p ${UPLOAD_DIR}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f --max-time 5 http://localhost:8088/health || exit 1

# Optimize Uvicorn workers based on memory constraints
ENV UVICORN_WORKERS=2

# Start command with optimized worker configuration
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "$UVICORN_WORKERS"]
