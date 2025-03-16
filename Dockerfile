FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
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
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    && python -m compileall -q /usr/local/lib/python3.9/site-packages

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . /app

# Environment variables (will be set via .env or docker-compose)
ENV PYTHONOPTIMIZE=1
ENV WHISPER_MODEL=base
ENV API_KEY=default_api_key
ENV PORT=8088
ENV HOST=0.0.0.0
ENV UPLOAD_DIR=/tmp/uploads
ENV DEBUG=false
ENV MAX_FILE_SIZE=52428800
ENV ALLOWED_ORIGINS=*
ENV RATE_LIMITS="10/minute,50/hour"
ENV CACHE_DIR=/tmp/whisper-cache

# Create directories and set permissions
RUN mkdir -p ${UPLOAD_DIR} /tmp/whisper-cache && \
    chown -R nobody:nogroup ${UPLOAD_DIR} /tmp/whisper-cache /app && \
    chmod -R 755 ${UPLOAD_DIR} /tmp/whisper-cache /app

# Switch to non-root user
USER nobody

# Expose port
EXPOSE ${PORT}

# Start command with optimized worker configuration
CMD ["python", "-OO", "server.py"]
