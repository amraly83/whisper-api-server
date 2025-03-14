# Whisper API Server

## Overview

The Whisper API Server is a robust solution for audio transcription using OpenAI's Whisper model. It provides an easy-to-use RESTful API interface for transcribing audio files with support for various response formats and advanced features like webhooks and caching.

## Features

- **Audio Transcription**: Transcribe audio files using the Whisper model.
- **Multiple Response Formats**: Supports JSON, text, SRT, VTT, and verbose JSON formats.
- **Webhook Support**: Register webhooks to receive transcription completion notifications.
- **Caching Mechanism**: Cache transcription results to improve performance for repeated requests.
- **Rate Limiting**: Protect your server from abuse with configurable rate limits.
- **Health Check Endpoint**: Monitor server health with a dedicated endpoint.
- **Security**: Authenticate API requests using API keys.

## Installation

### Prerequisites

- Python 3.8+
- Redis (for rate limiting and caching)
- Docker (optional, for containerized deployment)

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
   WHISPER_MODEL=small
   API_KEY=your_api_key_here
   PORT=8000
   HOST=0.0.0.0
   UPLOAD_DIR=/tmp/uploads
   MAX_FILE_SIZE=52428800  # 50MB
   DEBUG=false
   ALLOWED_ORIGINS=*
   RATE_LIMITS=10/minute,50/hour
   ```

4. **Start Redis**

   Ensure Redis is running on `localhost:6379`. You can start it using Docker:

   ```bash
   docker run -d --name redis -p 6379:6379 redis
   ```

5. **Run the Server**

   ```bash
   python server.py
   ```

   Alternatively, use Docker Compose:

   ```bash
   docker-compose up --build
   ```

## Usage

### Transcribe Audio

**Endpoint**: `POST /v1/audio/transcriptions`

**Parameters**:
- `model`: Model name (currently only `whisper-1` is supported).
- `file`: Audio file to transcribe.
- `response_format`: Response format (`json`, `text`, `srt`, `vtt`, `verbose_json`).
- `language`: Language code (optional).
- `prompt`: Initial prompt for transcription (optional).
- `temperature`: Sampling temperature (0.0 to 1.0).

**Example Request**:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
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

### Health Check

**Endpoint**: `GET /health`

**Example Request**:

```bash
curl -X GET "http://localhost:8000/health"
```

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
