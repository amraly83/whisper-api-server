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

# Create upload directory
RUN mkdir -p ${UPLOAD_DIR}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f --max-time 5 http://localhost:8088/health || exit 1

# Start command with optimized worker configuration
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "$UVICORN_WORKERS"]
