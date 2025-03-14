from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, Header, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import asyncio

import os
import shutil
import logging
from pathlib import Path
from typing import Any, List, Union, Optional

from datetime import timedelta
import secrets
import time

import numpy as np
import whisper
import torch

import uvicorn
import json
import base64
import tempfile
import hashlib

# File hashing
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Async transcription wrapper
async def async_transcribe(audio_path: str, **whisper_args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(worker_pool, transcribe, audio_path, **whisper_args)

# Configuration
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')
API_KEY = os.environ.get('API_KEY', secrets.token_hex(16))  # Generate a random API key if not provided
PORT = int(os.environ.get('PORT', 8088))
HOST = os.environ.get('HOST', '0.0.0.0')
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp')
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("whisper-api")

# Security
security = HTTPBearer()

# Global model and async processing
whisper_model = None
task_queue = Queue()
worker_pool = ThreadPoolExecutor(max_workers=4)

# Task tracking and caching
active_tasks = {}  # {task_id: {"status": "pending|processing|complete", "result": None}}
transcription_cache = {}  # {file_hash: transcription_result}
task_id_counter = 0

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
    global whisper_model
    # Load the ML model with CPU optimizations
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model with reduced memory footprint
    whisper_model = whisper.load_model(
        WHISPER_MODEL,
        device=device,
        in_memory=False,  # Reduce memory usage
        download_root="/tmp/whisper"  # Cache model files
    )
    
    # Enable CPU optimizations
    torch.set_num_threads(4)  # Limit CPU threads
    torch.backends.quantized.engine = 'qnnpack'  # Enable quantization
    logger.info("Whisper model loaded successfully")

    yield

    # Clean up the ML models and release the resources
    logger.info("Shutting down, releasing resources")
    del whisper_model
    whisper_model = None

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

# Generate unique task ID
def generate_task_id() -> str:
    global task_id_counter
    task_id_counter += 1
    return str(task_id_counter)

# Whisper transcription function
def transcribe(audio_path: str, task_id: str, **whisper_args):
    """Transcribe the audio file using whisper with chunked processing"""
    global whisper_model

    # Set configs & transcribe
    if whisper_args["temperature_increment_on_fallback"] is not None:
        whisper_args["temperature"] = tuple(
            np.arange(whisper_args["temperature"], 1.0 + 1e-6, whisper_args["temperature_increment_on_fallback"])
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]

    del whisper_args["temperature_increment_on_fallback"]

    logger.debug(f"Transcribing with args: {whisper_args}")
    start_time = time.time()

    # Update task status
    active_tasks[task_id]["status"] = "processing"
    
    # Process audio in chunks to reduce memory usage
    chunk_size = 30  # seconds
    audio = whisper.load_audio(audio_path)
    total_duration = len(audio) / whisper.audio.SAMPLE_RATE
    transcripts = []
    
    for start in range(0, int(total_duration), chunk_size):
        end = min(start + chunk_size, total_duration)
        logger.debug(f"Processing chunk {start}-{end} seconds")
        
        # Extract audio chunk
        start_sample = int(start * whisper.audio.SAMPLE_RATE)
        end_sample = int(end * whisper.audio.SAMPLE_RATE)
        audio_chunk = audio[start_sample:end_sample]
        
        # Transcribe chunk
        chunk_transcript = whisper_model.transcribe(
            audio_chunk,
            **whisper_args,
        )
        transcripts.append(chunk_transcript)
        
        # Clear memory
        del audio_chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Combine transcripts
    combined_transcript = {
        'text': ' '.join([t['text'] for t in transcripts]),
        'segments': [seg for t in transcripts for seg in t['segments']],
        'language': transcripts[0]['language'] if transcripts else None
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    
    # Log memory usage
    if torch.cuda.is_available():
        mem_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU memory usage: {mem_usage:.2f} MB")
    else:
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_info().rss / 1024**2
        logger.info(f"RAM usage: {mem_usage:.2f} MB")

    # Update task status and cache result
    active_tasks[task_id]["status"] = "complete"
    active_tasks[task_id]["result"] = combined_transcript

    return combined_transcript

@app.get("/v1/audio/transcriptions/{task_id}")
async def get_transcription_status(task_id: str, api_key: str = Depends(verify_api_key)):
    """Get the status or result of a transcription task"""
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID not found: {task_id}"
        )

    task = active_tasks[task_id]
    if task["status"] == "complete":
        return task["result"]
    else:
        return {"status": task["status"]}

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

    # Create temporary file and calculate hash
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.debug(f"File saved to temporary location: {temp_file_path}")
        file_hash = calculate_file_hash(temp_file_path)
        logger.info(f"File hash: {file_hash}")

        # Check if transcription is already cached
        if file_hash in transcription_cache:
            logger.info(f"Transcription found in cache for hash: {file_hash}")
            cached_result = transcription_cache[file_hash]
            
            # Format response based on requested format
            if response_format == 'text':
                return cached_result['text']
                
            elif response_format == 'srt':
                ret = ""
                for seg in cached_result['segments']:
                    td_s = timedelta(milliseconds=seg["start"]*1000)
                    td_e = timedelta(milliseconds=seg["end"]*1000)
                    
                    t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                    t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                    
                    ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"])
                ret += '\n'
                return ret
                
            elif response_format == 'vtt':
                ret = "WEBVTT\n\n"
                for seg in cached_result['segments']:
                    td_s = timedelta(milliseconds=seg["start"]*1000)
                    td_e = timedelta(milliseconds=seg["end"]*1000)
                    
                    t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                    t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                    
                    ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
                return ret
                
            elif response_format == 'verbose_json':
                cached_result.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
                cached_result.setdefault('duration', cached_result['segments'][-1]['end'])
                if cached_result['language'] == 'ja':
                    cached_result['language'] = 'japanese'
                return cached_result
                
            else:  # json (default)
                return {'text': cached_result['text']}

        # Generate task ID
        task_id = generate_task_id()
        active_tasks[task_id] = {"status": "pending", "result": None}
        logger.info(f"New transcription task created with ID: {task_id}")
        
        # Apply transcription settings
        settings = WHISPER_DEFAULT_SETTINGS.copy()
        settings['temperature'] = temperature
        if language is not None:
            settings['language'] = language
        if prompt is not None:
            settings['initial_prompt'] = prompt
        
        # Perform transcription asynchronously
        future = worker_pool.submit(transcribe, temp_file_path, task_id, **settings)

        def callback(fut):
            nonlocal file_hash
            if fut.exception():
                logger.error(f"Transcription task {task_id} failed: {fut.exception()}")
                active_tasks[task_id]["status"] = "error"
                active_tasks[task_id]["result"] = str(fut.exception())
            else:
                result = fut.result()
                transcription_cache[file_hash] = result
                logger.info(f"Transcription for task {task_id} completed and cached")

        future.add_done_callback(callback)
        
        # Return task ID for status checking
        return {"task_id": task_id, "status": "pending"}
            
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
    
    # Start the server
    uvicorn.run(
        "server:app", 
        host=HOST, 
        port=port, 
        log_level="debug" if DEBUG else "info"
    )

if __name__ == "__main__":
    main()
