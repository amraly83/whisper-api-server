from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from faster_whisper import WhisperModel
from io import BytesIO
import numpy as np
import soundfile as sf
import asyncio
import os
import json
import logging
from typing import List, Dict, Optional, Any, Iterator
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import time
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper API Server")
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses >1KB

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=settings.max_workers)

# Global model instance
whisper_model: Optional[WhisperModel] = None

# --- Performance Monitoring ---
@contextmanager
def timeit_context(name):
    start_time = time()
    yield
    elapsed = (time() - start_time) * 1000
    logger.info(f"{name} took {elapsed:.2f}ms")

# --- Model Management ---
@app.on_event("startup")
async def initialize_model():
    """Initialize and warm up the ASR model"""
    global whisper_model
    try:
        logger.info("Loading Whisper model...")
        with timeit_context("Model loading"):
            whisper_model = WhisperModel(
                settings.whisper_model_size,
                device=settings.device,
                compute_type=settings.compute_type,
                download_root="/models",  # Cache models to avoid redownloads
            )

        # Model warmup (critical for CUDA but useful for CPU too)
        logger.info("Warming up model...")
        with timeit_context("Model warmup"):
            dummy_audio = np.zeros((16000,), dtype=np.float32)  # 1s of silence
            list(whisper_model.transcribe(
                audio=dummy_audio,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_silence_duration_ms=500
                )
            ))
        
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request processing times"""
    start_time = time()
    response = await call_next(request)
    process_time = (time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
    return response

# --- Data Models ---
class TranscriptionResponse(BaseModel):
    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None

# --- Core Business Logic ---
def stream_generator(segments, info, response_format: str) -> Iterator[str]:
    """Generate streaming responses with optimized time-to-first-byte"""
    for segment in segments:
        chunk = process_segment(segment, response_format)
        if chunk:
            yield chunk
            
    # Finalization for JSON formats
    if response_format in ["sse_json", "verbose_json"]:
        yield json.dumps({"finished": True}) + "\n\n"

def process_segment(segment, response_format: str) -> Optional[str]:
    """Process a single transcript segment into the requested format"""
    # Minimum text length to send (reduces small chunks)
    min_length = 4 if response_format in ["text", "vtt", "srt"] else 1
    
    if len(segment.text) < min_length:
        return None

    return {
        "sse_json": lambda: f"data: {json.dumps({'text': segment.text})}\n\n",
        "verbose_json": lambda: f"""data: {json.dumps({
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'confidence': segment.avg_logprobt
        })}\n\n""",
        "text": lambda: f"{segment.text} ",
        "srt": lambda: format_subtitle(segment, "srt"),
        "vtt": lambda: format_subtitle(segment, "vtt")
    }.get(response_format, lambda: None)()

def format_subtitle(segment, format_type: str) -> str:
    """Format subtitle content with proper timestamps"""
    return (
        f"{format_timestamp(segment.start, format_type)} --> "
        f"{format_timestamp(segment.end, format_type)}\n"
        f"{segments.text}\n\n"
    )

def format_timestamp(seconds: float, format_type: str) -> str:
    """Optimized timestamp formatting"""
    ms = int((seconds - int(seconds)) * 1000)
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    
    return {
        "srt": f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}",
        "vtt": f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"
    }[format_type]

# --- API Endpoints ---
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str = Form(default=None),
    prompt: str = Form(default=None),
    response_format: str = Form(default="sse_json"),
    temperature: float = Form(default=0),
    stream: bool = Form(default=False),
):
    """Optimized transcription endpoint supporting streaming"""
    # Validate inputs
    if response_format not in ["sse_json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(status_code=400, detail="Unsupported response format")

    try:
        # In-memory audio processing (no disk I/O)
with timeit_context("Audio processing"):
    content = await file.read()
    
    try:
        # Use soundfile with proper error handling
        with BytesIO(content) as audio_buffer:
            audio_data, sample_rate = sf.read(
                audio_buffer,
                dtype='float32',
                always_2d=True,
                fill_value=0.0  # Handle truncated files
            )
    except sf.LibsndfileError as e:
        logger.error(f"Invalid audio file: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid audio file format")
        
    # Convert to mono and validate content
    if audio_data.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
        
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Debug: Verify audio content
    logger.info(f"Audio loaded: {audio_data.shape} samples, {np.max(audio_data):.2f} max amplitude")

        # Configure transcription parameters
        transcribe_params = dict(
            language=language,
            initial_prompt=prompt,
            temperature=0 if stream else temperature,
            beam_size=1 if stream else settings.beam_size,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=500
            )
        )

        # Run transcription in thread pool
        with timeit_context("Transcription"):
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                executor,
                lambda: whisper_model.transcribe(
                    audio=audio_data,  # Directly pass numpy array
                    **transcribe_params
                )
            )
            segments, info = await future

        # Streaming response
        if stream:
            return StreamingResponse(
                stream_generator(segments, info, response_format),
                media_type="text/event-stream" if "json" in response_format else "text/plain"
            )

        # Non-streaming response
        return format_non_streaming_response(segments, info, response_format)

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
def format_non_streaming_response(segments, info, response_format: str):
    """Format complete transcript response"""
    text = "".join(seg.text for seg in segments)
    
    if response_format == "text":
        return text
    elif response_format == "sse_json":
        return {"text": text}
    elif response_format == "verbose_json":
        return {
            "text": text,
            "segments": [{
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.avg_logprob
            } for seg in segments],
            "language": info.language
        }
    # Handle SRT/VTT formats similarly...

# --- Health Checks ---
@app.get("/")
async def root():
    return {"status": "running", "model": settings.whisper_model_size}

@app.get("/health")
async def health():
    return {"status": "healthy", "workers": executor._max_workers}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8083,
        server_header=False,
        timeout_keep_alive=60  # Allow HTTP keep-alives
    )
