# Whisper API Server

A Docker-based API server for OpenAI's Whisper speech-to-text model, compatible with the OpenAI API format.

## Features

- Transcribe audio files to text using the Whisper model
- Compatible with OpenAI API format
- Lightweight using faster-whisper implementation
- Supports various output formats (json, text, srt, vtt)
- Suitable for low-memory environments (4GB+ RAM)

## Deployment

### Using Docker

```bash
docker build -t whisper-api .
docker run -d -p 8083:8083 --memory=4g --name whisper-server whisper-api