from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from redis import Redis
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import asyncio

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from datetime import timedelta
import time
import hmac  # For webhook signature verification
import requests  # For webhook delivery
from pydantic import BaseModel, HttpUrl  # For request validation
from fastapi import Query  # For query parameter validation

import whisper
import torch

import uvicorn
import json
import tempfile
import hashlib

# Setup basic logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

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

# Enhanced Configuration
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    print("[ERROR] API_KEY environment variable is required")
    raise SystemExit("API_KEY environment variable is required")

PORT = os.environ.get('PORT', '8000')
try:
    PORT = int(PORT)
    if PORT < 1024 or PORT > 65535:
        raise ValueError(f"Invalid port number: {PORT}")
except ValueError as e:
    print(f"[ERROR] Invalid PORT configuration: {e}")
    raise SystemExit(f"Invalid PORT configuration: {e}")

HOST = os.environ.get('HOST', '0.0.0.0')
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp/uploads')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Additional security settings
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
RATE_LIMITS = os.environ.get('RATE_LIMITS', "10/minute,50/hour").split(',')


# Enhanced logging setup
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        req_body = b""
        
        async def receive_wrapper():
            nonlocal req_body
            message = await receive()
            if message["type"] == "http.request":
                req_body += message.get("body", b"")
            return message
        
        response_status = None
        res_body = b""
        
        async def send_wrapper(message):
            nonlocal response_status, res_body
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                res_body += message.get("body", b"")
            await send(message)
        
        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            raise
        finally:
            duration = int((time.time() - start_time) * 1000)
            log_data = {
                "method": scope["method"],
                "path": scope["path"],
                "status_code": response_status,
                "duration_ms": duration,
                "request_size": len(req_body),
                "response_size": len(res_body),
            }
            logger.info(f"API Request: {json.dumps(log_data)}")
            
            # Log errors separately
            if response_status and response_status >= 400:
                error_data = {
                    "error": {
                        "status_code": response_status,
                        "path": scope["path"],
                        "method": scope["method"],
                        "response": res_body.decode(errors="ignore"),
                        "duration_ms": duration
                    }
                }
                logger.error(f"API Error: {json.dumps(error_data)}")

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("whisper-api")

# Performance metrics
performance_metrics = {
    "total_requests": 0,
    "success_count": 0,
    "error_count": 0,
    "average_response_time": 0,
    "response_times": []
}

def update_metrics(status_code: int, duration: float):
    """Update performance metrics"""
    performance_metrics["total_requests"] += 1
    performance_metrics["response_times"].append(duration)
    performance_metrics["average_response_time"] = sum(performance_metrics["response_times"]) / len(performance_metrics["response_times"])
    
    if 200 <= status_code < 400:
        performance_metrics["success_count"] += 1
    else:
        performance_metrics["error_count"] += 1

# Security setup
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials
# Rate limiting setup
redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
limiter = Limiter(key_func=get_remote_address, storage_uri="redis://redis:6379")

# Global model and async processing
whisper_model = None
task_queue = Queue()
worker_pool = ThreadPoolExecutor(max_workers=4)

# Rate limit configuration (adjust based on server capacity)
RATE_LIMITS = ["10/minute", "50/hour"]  # Max 10 requests per minute, 50 per hour

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
    "fp16": False  # Ensure FP16 is disabled
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

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Authentication dependency

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
            whisper_args["temperature"] = [
                whisper_args["temperature"] + i * whisper_args["temperature_increment_on_fallback"]
                for i in range(int((1.0 - whisper_args["temperature"]) / whisper_args["temperature_increment_on_fallback"]) + 1)
            ]
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
async def get_transcription_status(
    task_id: str, 
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    api_key: str = Depends(verify_api_key)
):
    """
    Get the status or result of a transcription task with pagination
    """
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID not found: {task_id}"
        )

    task = active_tasks[task_id]
    if task["status"] != "complete":
        return {"status": task["status"]}
    
    # Paginate results
    result = task["result"]
    total_segments = len(result['segments'])
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_segments = result['segments'][start_idx:end_idx]
    has_more = end_idx < total_segments
    
    return {
        "text": result['text'],
        "language": result['language'],
        "total_segments": total_segments,
        "page": page,
        "page_size": page_size,
        "has_more": has_more,
        "segments": paginated_segments
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Webhook model
class Webhook(BaseModel):
    url: HttpUrl
    secret: Optional[str] = None

webhooks = {}  # {task_id: {"url": webhook_url, "secret": webhook_secret}}

@app.post('/v1/webhooks')
@limiter.limit(RATE_LIMITS[1])  # Apply less strict rate limit for webhooks
async def register_webhook(
    request: Request,
    webhook: Webhook,
    task_id: str = Query(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Register a webhook for transcription completion notifications
    """
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID not found: {task_id}"
        )
    
    webhooks[task_id] = {
        "url": webhook.url,
        "secret": webhook.secret
    }
    
    return {"message": "Webhook registered successfully"}

def send_webhook_notification(task_id: str):
    """Send webhook notification in background"""
    if task_id not in webhooks:
        return
    
    webhook = webhooks[task_id]
    task = active_tasks[task_id]
    
    headers = {}
    if webhook['secret']:
        signature = hmac.new(
            webhook['secret'].encode(),
            json.dumps(task).encode(),
            hashlib.sha256
        ).hexdigest()
        headers['X-Webhook-Signature'] = signature
    
    try:
        requests.post(
            webhook['url'],
            json=task,
            headers=headers,
            timeout=5
        )
    except Exception as e:
        logger.error(f"Webhook delivery failed for task {task_id}: {str(e)}")

@app.post('/v1/audio/transcriptions')
@limiter.limit(RATE_LIMITS.split(',')[0])  # Apply rate limiting to endpoint
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
        
    logger.info(f"Received transcription request for file: {file.filename}")
    
    # Check remaining rate limit
    current_limit = limiter.current_limit
    remaining = current_limit.get_retry_after() if current_limit else None
    logger.debug(f"Rate limit remaining: {remaining}")
    
    # Existing code...
    
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
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.debug(f"File saved to temporary location: {temp_file_path}")
        logger.debug(f"Checking if file exists at {temp_file_path}: {os.path.exists(temp_file_path)}")
        if not os.path.exists(temp_file_path):
            logger.error(f"Temporary file not found at {temp_file_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Temporary file not found at {temp_file_path}"
            )
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
        
        # Check if webhook is provided
        if webhook_url:
            webhook_task = asyncio.create_task(
                async_transcribe(temp_file_path, **settings)
            )
            
            # Register webhook
            webhooks[task_id] = {
                "url": webhook_url,
                "secret": webhook_secret
            }
            
            # Process transcription asynchronously
            def process_transcription():
                try:
                    result = asyncio.run(webhook_task)
                    transcription_cache[file_hash] = result
                    
                    # Update task status
                    active_tasks[task_id]["status"] = "complete"
                    active_tasks[task_id]["result"] = result
                    
                    # Send webhook notification
                    send_webhook_notification(task_id)
                except Exception as e:
                    logger.error(f"Async transcription failed: {str(e)}")
                    active_tasks[task_id]["status"] = "failed"
                    active_tasks[task_id]["error"] = str(e)
                    
                    # Send failure notification
                    send_webhook_notification(task_id)
            
            # Run transcription in background thread
            worker_pool.submit(process_transcription)
            
            return {
                "message": "Transcription started",
                "task_id": task_id,
                "status": "processing",
                "webhook_registered": True
            }
        
        # Perform synchronous transcription if no webhook
        result = transcribe(temp_file_path, task_id, **settings)
        transcription_cache[file_hash] = result
        
        # Format response based on requested format
        if response_format == 'text':
            return result['text']
            
        elif response_format == 'srt':
            ret = ""
            for seg in result['segments']:
                td_s = timedelta(milliseconds=seg["start"]*1000)
                td_e = timedelta(milliseconds=seg["end"]*1000)
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"])
            ret += '\n'
            return ret
            
        elif response_format == 'vtt':
            ret = "WEBVTT\n\n"
            for seg in result['segments']:
                td_s = timedelta(milliseconds=seg["start"]*1000)
                td_e = timedelta(milliseconds=seg["end"]*1000)
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
            return ret
            
        elif response_format == 'verbose_json':
            result.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
            result.setdefault('duration', result['segments'][-1]['end'])
            if result['language'] == 'ja':
                result['language'] = 'japanese'
            return result
            
        else:  # json (default)
            return {'text': result['text']}
            
    except Exception as e:
        logger.error(f"Error processing transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing transcription: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            logger.debug(f"Checking if temporary file exists: {os.path.exists(temp_file_path)}")
            if os.path.exists(temp_file_path):
                logger.debug(f"Temporary file found: {temp_file_path}")
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Temporary file removed: {temp_file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {temp_file_path}, error: {str(e)}")
            else:
                logger.debug(f"Temporary file not found: {temp_file_path}")

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom rate limit exceeded handler
    """
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded", "retry_after": exc.retry_after},
    )

def main():
    # Print the API key to console
    logger.info(f"API Key: {API_KEY}")
    
    # Initialize rate limiter
    limiter.redis = redis_client
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
