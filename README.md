# Whisper API Server

## Overview

The Whisper API Server is a robust solution for audio transcription using OpenAI's Whisper model. It provides an easy-to-use RESTful API interface for transcribing audio files with support for various response formats and advanced features like webhooks and caching.

## Features

- **Audio Transcription**: Transcribe audio files using the Whisper model
- **Chunked Processing**: Process long audio files in chunks for better memory management
- **Multiple Response Formats**: Supports JSON, text, SRT, VTT, and verbose JSON formats
- **Webhook Support**: Register webhooks to receive transcription completion notifications
- **Smart Caching**: Cache transcription results and model files for improved performance
- **Rate Limiting**: Protect your server from abuse with configurable rate limits
- **Resource Management**: Automatic garbage collection and model unloading for optimal memory usage
- **Health Check Endpoint**: Monitor server health with a dedicated endpoint
- **Security**: Authenticate API requests using API keys

## Installation

### Prerequisites

- Python 3.8+
- Redis (for rate limiting and caching)
- Docker (optional, for containerized deployment)
- At least 2GB RAM (3GB recommended for optimal performance)
- CUDA-compatible GPU (optional, for faster processing)

### Key Dependencies
- FastAPI (>= 0.100.0) - Modern web framework
- faster-whisper - Optimized Whisper implementation
- ctranslate2 (>= 3.16.0) - Neural machine translation
- onnxruntime - Machine learning inference
- torch - Deep learning framework
- Other dependencies are listed in requirements.txt

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/whisper-api-server.git
   cd whisper-api-server
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**

   Create a `.env` file in the root directory with the following variables:

   ```env
   # Server Configuration
   PORT=8088
   HOST=0.0.0.0
   DEBUG=false
   API_KEY=your_api_key_here

   # Model Configuration
   WHISPER_MODEL=base
   CHUNK_SIZE_SECONDS=10
   CHUNK_OVERLAP=0.5
   MAX_CONCURRENT_TASKS=2
   MODEL_UNLOAD_IDLE_MINUTES=2

   # File Handling
   UPLOAD_DIR=/tmp/uploads
   MAX_FILE_SIZE=52428800  # 50MB

   # Security
   ALLOWED_ORIGINS=*
   RATE_LIMITS=10/minute,50/hour

   # Resource Management
   GC_FREQUENCY=2  # Garbage collection frequency in minutes
   ```

4. **Start the Server**

   Using Docker Compose (recommended):

   ```bash
   docker-compose up --build
   ```

   The server will be available at `http://localhost:8088`

## Resource Management

The server includes several features for optimal resource management:

- **Memory Limits**: Container is limited to 3GB RAM with 512MB minimum reservation
- **CPU Limits**: Uses up to 2 CPU cores with 1 core reserved
- **Auto-cleanup**: Garbage collection runs every 2 minutes
- **Model Management**: Unloads model after 2 minutes of inactivity
- **Redis Memory**: Limited to 384MB with LRU eviction policy

## Volume Persistence

The following data is persisted across container restarts:

- `/app/uploads`: Temporary audio file storage
- `/app/whisper_cache`: Model cache storage
- Redis data: Cached transcriptions and rate limit data

## Usage

### Transcribe Audio

**Endpoint**: `POST /v1/audio/transcriptions`

**Parameters**:
- `model`: Model name (currently only `whisper-1` is supported)
- `file`: Audio file to transcribe
- `response_format`: Response format (`json`, `text`, `srt`, `vtt`, `verbose_json`)
- `language`: Language code (optional)
- `prompt`: Initial prompt for transcription (optional)
- `temperature`: Sampling temperature (0.0 to 1.0)

**Example Request**:

```bash
curl -X POST "http://localhost:8088/v1/audio/transcriptions" \
     -H "Authorization: Bearer your_api_key_here" \
     -F "model=whisper-1" \
     -F "file=@/path/to/audio/file.mp3" \
     -F "response_format=json"
```

### Register Webhook

**Endpoint**: `POST /v1/webhooks`

**Parameters**:
- `task_id`: Task ID of the transcription.
- `url`: Webhook URL.
- `secret`: Webhook secret (optional).

**Example Request**:

```bash
curl -X POST "http://localhost:8000/v1/webhooks" \
     -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{
           "task_id": "1",
           "url": "https://example.com/webhook",
           "secret": "your_webhook_secret"
         }'
```

### Get Transcription Status

**Endpoint**: `GET /v1/audio/transcriptions/{task_id}`

**Parameters**:
- `task_id`: Task ID of the transcription.
- `page`: Page number for paginated results.
- `page_size`: Number of segments per page.

**Example Request**:

```bash
curl -X GET "http://localhost:8000/v1/audio/transcriptions/1?page=1&page_size=50" \
     -H "Authorization: Bearer your_api_key_here"
```

## Performance Optimization

### Chunk Processing
- Long audio files are automatically split into 10-second chunks
- 0.5-second overlap between chunks ensures smooth transitions
- Parallel processing of chunks with maximum 2 concurrent tasks

### Caching Strategy
- Model files are cached in persistent storage
- Transcription results are cached in Redis
- Redis uses LRU eviction when reaching 384MB memory limit

## Monitoring

### Health Check

**Endpoint**: `GET /health`

Returns server status including:
- System resources (CPU, Memory, Disk usage)
- Redis connection status
- Model status
- Active tasks count

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `413`: Payload Too Large (file size exceeds limit)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error (processing failed)

## API Documentation

For detailed API documentation, refer to the [API Reference](docs/api-reference.md).

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainers at your-email@example.com.
