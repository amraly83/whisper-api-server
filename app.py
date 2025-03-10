from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from faster_whisper import WhisperModel
from io import BytesIO
import numpy as np
import soundfile as sf
import asyncio
import os
import json
import logging
import hashlib
import redis
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import time
from cachetools import TTLCache
from config import settings
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis for rate limiting
redis_client = redis.StrictRedis.from_url(settings.redis_url)

# Setup application
app = FastAPI(title="Whisper API Server", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Caching and Rate Limiting
response_cache = TTLCache(maxsize=1000, ttl=settings.cache_ttl)
rate_limit_cache = TTLCache(maxsize=10000, ttl=60)

# Thread pool and model initialization
executor = ThreadPoolExecutor(max_workers=settings.max_workers)
whisper_model: Optional[WhisperModel] = None

# --- Middlewares ---
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Request validation
    if request.method != "GET" and request.url.path == "/v1/audio/transcriptions":
        content_type = request.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return JSONResponse(
                content={"detail": "Unsupported media type"},
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            )

    # Rate limiting
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    rate_key = f"rate_limit:{client_ip}"
    
    current_count = redis_client.incr(rate_key)
    if current_count > settings.rate_limit:
        return JSONResponse(
            content={"detail": "Rate limit exceeded"},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )
    if current_count == 1:
        redis_client.expire(rate_key, 60)
        
    return await call_next(request)

# --- Security Dependencies ---
async def validate_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True

# --- Model Initialization ---
@app.on_event("startup")
async def initialize_model():
    global whisper_model
    try:
        logger.info("Initializing Whisper model...")
        whisper_model = WhisperModel(
            settings.whisper_model_size,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root="/models"
        )
        # Warmup model
        dummy_audio = np.zeros((16000,), dtype=np.float32)
        list(whisper_model.transcribe(dummy_audio))
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

# --- Audio Validation ---
def validate_audio(input_data: bytes) -> np.ndarray:
    try:
        data, sr = sf.read(BytesIO(input_data), dtype='float32')
        if data.size == 0:
            raise ValueError("Audio file is empty")
        return np.mean(data, axis=1) if data.ndim > 1 else data
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid audio file")

def convert_to_wav(input_data: bytes) -> bytes:
    """Converts audio data to 16kHz mono WAV format using ffmpeg."""
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-i", "pipe:0",  # Read from stdin
                "-ar", "16000",  # Set sample rate to 16kHz
                "-ac", "1",  # Set channels to mono
                "-f", "wav",  # Output format: WAV
                "pipe:1"  # Write to stdout
            ],
            input=input_data,
            capture_output=True,
            check=True
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
        raise HTTPException(status_code=500, detail="Audio conversion failed")

# --- Core Operations ---
class TranscriptionResponse(BaseModel):
    text: str
    duration: float
    language: str
    segments: Optional[List[Dict[str, Any]]] = None

def generate_cache_key(content: bytes, params: dict) -> str:
    return hashlib.sha256(content + json.dumps(params).encode()).hexdigest()

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    _: bool = Depends(validate_api_key)
):
    # Input validation
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(400, "Unsupported response format")

    try:
        content = await file.read()
        params = {
            "language": language,
            "prompt": prompt,
            "temperature": temperature,
            "format": response_format
        }
        cache_key = generate_cache_key(content, params)
        
        # Check cache
        if cache_key in response_cache:
            logger.info(f"Cache hit for {cache_key[:8]}")
            return response_cache[cache_key]

        # Process audio
        with timeit_context("audio_processing"):
            # Convert to 16kHz mono WAV
            wav_data = convert_to_wav(content)
            audio_data = validate_audio(wav_data)
            duration = len(audio_data) / 16000  # Assumes 16kHz sample rate
            if duration > settings.max_duration:
                raise HTTPException(400, f"Audio exceeds {settings.max_duration}s limit")

        # Transcribe
        with timeit_context("transcription"):
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                executor,
                lambda: whisper_model.transcribe(
                    audio=audio_data,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 250},
                ),
            )

        # Format response
        response = {
            "text": "".join(seg.text for seg in segments),
            "duration": duration,
            "language": info.language
        }
        
        if response_format == "verbose_json":
            response["segments"] = [{
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": round(seg.avg_logprob, 2)
            } for seg in segments]

        # Cache result
        response_cache[cache_key] = response
        return response

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(500, "Processing failed") from e

# --- Health Endpoints ---
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "workers": executor._max_workers}

@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Whisper API", "status": "running"}

# --- Utility Functions ---
@contextmanager
def timeit_context(name: str):
    start = time()
    try:
        yield
    finally:
        elapsed = (time() - start) * 1000
        logger.info(f"{name}: {elapsed:.2f}ms")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8083")),
        log_config=None
    )
