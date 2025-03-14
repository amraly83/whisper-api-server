from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, Header, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

import os
import shutil
import logging
from pathlib import Path
from typing import Any, List, Union, Optional

from datetime import timedelta
import secrets
import time
import multiprocessing

import numpy as np
import whisper
import torch
import warnings

import uvicorn
import json
import base64
import tempfile

# Configuration
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')
API_KEY = os.environ.get('API_KEY', secrets.token_hex(16))  # Generate a random API key if not provided
PORT = int(os.environ.get('PORT', 8088))
HOST = os.environ.get('HOST', '0.0.0.0')
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp')
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', max(2, multiprocessing.cpu_count() - 1)))
USE_GPU = os.environ.get('USE_GPU', 'false').lower() == 'true'
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 30))  # seconds

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("whisper-api")

# Filter out unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Security
security = HTTPBearer()

# Global model and async processing
whisper_model = None
worker_pool = None

# Whisper default settings
WHISPER_DEFAULT_SETTINGS = {
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, worker_pool
    # Determine device
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on {device}")
    
    # Optimize based on the device
    if device == "cpu":
        # Set torch parameters for CPU optimization
        torch.set_num_threads(NUM_WORKERS)  # Use available cores
        torch.set_num_interop_threads(NUM_WORKERS)
        torch.backends.quantized.engine = 'qnnpack'  # Enable quantization
    else:
        # For GPU, make sure to enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load model with appropriate optimization flags
    whisper_model = whisper.load_model(
        WHISPER_MODEL,
        device=device,
        in_memory=True,         # Keep model in memory for faster inference
        download_root="/tmp/whisper"  # Cache model files
    )
    
    # Create worker pool based on device capabilities
    if device == "cpu":
        worker_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        logger.info(f"Created ThreadPoolExecutor with {NUM_WORKERS} workers")
    else:
        worker_pool = ThreadPoolExecutor(max_workers=4)  # Fewer workers for GPU to avoid memory issues
        logger.info(f"Created ThreadPoolExecutor with 4 workers for GPU")
    
    logger.info("Whisper model loaded successfully")
    yield

    # Clean up resources
    logger.info("Shutting down, releasing resources")
    worker_pool.shutdown()
    del whisper_model
    whisper_model = None
    if device == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(
    title="Whisper API",
    description="API for audio transcription using Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication scheme",
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return credentials.credentials

# Audio file preprocessing
def preprocess_audio(audio_path: str):
    """Load and preprocess audio file for faster transcription"""
    audio = whisper.load_audio(audio_path)
    
    # Normalize audio (improves transcription quality & speed)
    if not np.isclose(np.max(np.abs(audio)), 1.0):
        audio = audio / np.max(np.abs(audio))
    
    return audio

# Process a single audio chunk
def process_chunk(args):
    """Process a single audio chunk"""
    chunk, whisper_args = args
    global whisper_model
    
    # Process chunk
    try:
        with torch.no_grad():  # Disable gradient calculation for inference
            result = whisper_model.transcribe(chunk, **whisper_args)
        return result
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return {"text": "", "segments": []}

# Whisper transcription function
async def transcribe(audio_path: str, **whisper_args):
    """Transcribe the audio file using whisper with parallel chunk processing"""
    global whisper_model, worker_pool
    
    # Set configs
    if whisper_args.get("temperature_increment_on_fallback") is not None:
        whisper_args["temperature"] = tuple(
            np.arange(whisper_args["temperature"], 1.0 + 1e-6, whisper_args["temperature_increment_on_fallback"])
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]
    
    if "temperature_increment_on_fallback" in whisper_args:
        del whisper_args["temperature_increment_on_fallback"]
    
    logger.debug(f"Transcribing with args: {whisper_args}")
    start_time = time.time()
    
    # Preprocess audio
    audio = preprocess_audio(audio_path)
    total_duration = len(audio) / whisper.audio.SAMPLE_RATE
    
    # Process audio in parallel chunks to improve performance
    chunks = []
    chunk_size_samples = int(CHUNK_SIZE * whisper.audio.SAMPLE_RATE)
    
    # Create chunks
    for start_sample in range(0, len(audio), chunk_size_samples):
        end_sample = min(start_sample + chunk_size_samples, len(audio))
        chunks.append((audio[start_sample:end_sample], whisper_args))
    
    logger.debug(f"Split audio into {len(chunks)} chunks of {CHUNK_SIZE} seconds each")
    
    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    try:
        # Submit all chunks to the worker pool and gather results
        chunk_results = await loop.run_in_executor(
            None,  # Use the default executor
            lambda: list(worker_pool.map(process_chunk, chunks))
        )
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}", exc_info=True)
        raise
    
    # Combine results
    combined_text = " ".join([result["text"] for result in chunk_results if result])
    
    # Adjust segment timestamps across chunks
    all_segments = []
    offset = 0
    for i, result in enumerate(chunk_results):
        if not result or "segments" not in result:
            continue
            
        for segment in result["segments"]:
            segment["start"] += offset
            segment["end"] += offset
            all_segments.append(segment)
        
        offset += CHUNK_SIZE
    
    # Create final transcript
    combined_transcript = {
        'text': combined_text,
        'segments': all_segments,
        'language': chunk_results[0].get('language') if chunk_results and chunk_results[0] else None
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds ({total_duration:.2f}s audio)")
    
    return combined_transcript

@app.get('/')
async def root():
    return {
        "name": "Whisper API",
        "description": "API for audio transcription using Whisper",
        "version": "1.0.0",
        "endpoints": {
            "/v1/models": "GET - List available models",
            "/v1/audio/transcriptions": "POST - Transcribe audio files",
        }
    }

@app.get('/v1/models')
async def v1_models():
    logger.debug("Returning available models")
    content = {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "whisper-api"
            }
        ]
    }

    return JSONResponse(
        content=content,
        status_code=200
    )

@app.post('/v1/audio/transcriptions')
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"Received transcription request for file: {file.filename}")
    
    # Validate model
    if model != "whisper-1":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model}. Only 'whisper-1' is supported"
        )
    
    # Validate response_format
    if response_format is None:
        response_format = 'json'
    if response_format not in ['json', 'text', 'srt', 'verbose_json', 'vtt']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported response_format: {response_format}"
        )
    
    # Validate temperature
    if temperature is None:
        temperature = 0.0
    if temperature < 0.0 or temperature > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid temperature: {temperature}. Must be between 0.0 and 1.0"
        )
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.debug(f"File saved to temporary location: {temp_file_path}")
        
        # Apply transcription settings
        settings = WHISPER_DEFAULT_SETTINGS.copy()
        settings['temperature'] = temperature
        if language is not None:
            settings['language'] = language
        if prompt is not None:
            settings['initial_prompt'] = prompt
        
        # Perform transcription
        transcript = await transcribe(audio_path=temp_file_path, **settings)
        
        # Format response based on requested format
        if response_format == 'text':
            return transcript['text']
            
        elif response_format == 'srt':
            ret = ""
            for i, seg in enumerate(transcript['segments']):
                td_s = timedelta(seconds=seg["start"])
                td_e = timedelta(seconds=seg["end"])
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += '{}\n{} --> {}\n{}\n\n'.format(i+1, t_s, t_e, seg["text"])
            ret += '\n'
            return ret
            
        elif response_format == 'vtt':
            ret = "WEBVTT\n\n"
            for seg in transcript['segments']:
                td_s = timedelta(seconds=seg["start"])
                td_e = timedelta(seconds=seg["end"])
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
            return ret
            
        elif response_format == 'verbose_json':
            transcript.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
            if transcript['segments']:
                transcript.setdefault('duration', transcript['segments'][-1]['end'])
            if transcript['language'] == 'ja':
                transcript['language'] = 'japanese'
            return transcript
            
        else:  # json (default)
            return {'text': transcript['text']}
        
    except Exception as e:
        logger.error(f"Error processing transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing transcription: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temporary file removed: {temp_file_path}")

# Health check endpoint
@app.get('/health')
async def health_check():
    """Health check endpoint to verify server status"""
    return {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "device": whisper_model.device if whisper_model else "not loaded",
        "workers": NUM_WORKERS,
    }

def main():
    # Print the API key to console
    logger.info(f"API Key: {API_KEY}")
    
    # Validate PORT
    try:
        port = int(PORT)
        if port < 1024 or port > 65535:
            raise ValueError(f"Port must be between 1024 and 65535, got {port}")
    except ValueError as e:
        logger.error(f"Invalid PORT value: {PORT} - {str(e)}")
        raise SystemExit(1)
        
    logger.info(f"Starting server on {HOST}:{port}")
    logger.info(f"Using {NUM_WORKERS} workers and {'GPU' if USE_GPU else 'CPU'} for processing")
    
    # Start the server
    uvicorn.run(
        "server:app", 
        host=HOST, 
        port=port, 
        log_level="debug" if DEBUG else "info",
        workers=1  # FastAPI instance should use one worker since we handle concurrency internally
    )

if __name__ == "__main__":
    main()