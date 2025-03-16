from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from redis import Redis
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, HttpUrl, Field, validator
import asyncio
import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from datetime import timedelta
import time
import hmac
import requests
import tempfile
import hashlib
import uvicorn
import json
import psutil
import uuid
import gc
import subprocess
from enum import Enum
import mimetypes
import numpy as np

try:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    import torch
except ImportError:
    print("Error: Required ML libraries not installed.")
    print("Install with: pip install faster-whisper ctranslate2 onnxruntime")
    exit(1)

# Enhanced logging with structured format
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S,%f")[:-3],
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

# Setup logging early
def setup_logging(log_dir="/var/log/whisper-api", debug=False):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "whisper-api.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(file_handler)
    
    return logging.getLogger("whisper-api")

# Configuration with validation
class AppConfig:
    def __init__(self):
        self.WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')
        self.API_KEY = os.environ.get('API_KEY')
        
        if not self.API_KEY:
            raise ValueError("API_KEY environment variable is required")
        
        try:
            self.PORT = int(os.environ.get('PORT', '8088'))
            if self.PORT < 1024 or self.PORT > 65535:
                raise ValueError(f"Invalid port number: {self.PORT}")
        except ValueError as e:
            raise ValueError(f"Invalid PORT configuration: {e}")
        
        self.HOST = os.environ.get('HOST', '0.0.0.0')
        self.UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp/uploads')
        self.MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        
        self.DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
        self.ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
        self.RATE_LIMITS = os.environ.get('RATE_LIMITS', "10/minute,50/hour").split(',')
        
        # New configurations
        self.MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', '4'))
        self.CACHE_DIR = os.environ.get('CACHE_DIR', '/tmp/whisper-cache')
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        self.MEMORY_THRESHOLD = float(os.environ.get('MEMORY_THRESHOLD', '0.8'))  # 80% memory threshold
        self.ALLOWED_AUDIO_TYPES = os.environ.get('ALLOWED_AUDIO_TYPES', 
                                                 'audio/mpeg,audio/wav,audio/x-wav,audio/ogg').split(',')
        
        # Logging config
        self.LOG_DIR = os.environ.get('LOG_DIR', '/var/log/whisper-api')
        
        # Redis configuration
        self.REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
        
        # Task cleanup configuration
        self.TASK_RETENTION_HOURS = int(os.environ.get('TASK_RETENTION_HOURS', '24'))
        
        # Model management
        self.MODEL_UNLOAD_IDLE_MINUTES = int(os.environ.get('MODEL_UNLOAD_IDLE_MINUTES', '5'))  # More aggressive unloading for 4GB RAM
        
        # Chunking configuration - optimized for real-time
        self.CHUNK_SIZE_SECONDS = int(os.environ.get('CHUNK_SIZE_SECONDS', '10'))  # Reduced for faster processing
        self.CHUNK_OVERLAP = float(os.environ.get('CHUNK_OVERLAP', '0.5'))  # Reduced overlap
        self.MAX_AUDIO_DURATION = int(os.environ.get('MAX_AUDIO_DURATION', '300'))  # 5 minutes max
        
        # Memory optimization
        self.MEMORY_THRESHOLD = float(os.environ.get('MEMORY_THRESHOLD', '0.85'))
        self.GC_FREQUENCY = int(os.environ.get('GC_FREQUENCY', '2'))  # Run GC every N chunks
        self.MODEL_UNLOAD_IDLE_MINUTES = int(os.environ.get('MODEL_UNLOAD_IDLE_MINUTES', '2'))  # More aggressive unloading

# Request ID middleware for tracking requests
class RequestIDMiddleware:
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            logger.error(f"Request {request_id} failed", exc_info=True)
            raise

# Enhanced logging middleware
class LoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Add request_id to log records
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record
        logging.setLogRecordFactory(record_factory)
        
        logger.info(f"Request started: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration:.3f}s"
            )
            
            # Update metrics
            update_metrics(response.status_code, duration)
            
            return response
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"duration={duration:.3f}s",
                exc_info=True
            )
            raise
        finally:
            # Reset log record factory
            logging.setLogRecordFactory(old_factory)
            
            # Log memory usage periodically
            if random.random() < 0.1:  # 10% sampling
                log_system_metrics()

# Models and type definitions
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

class Webhook(BaseModel):
    url: HttpUrl
    secret: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/webhook",
                "secret": "your_webhook_secret"
            }
        }

class TranscriptionRequest(BaseModel):
    model: str = Field(..., description="Model to use for transcription")
    response_format: Optional[str] = Field(None, description="Response format (json, text, srt, vtt, verbose_json)")
    language: Optional[str] = Field(None, description="Language code for transcription")
    prompt: Optional[str] = Field(None, description="Initial prompt for transcription")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature (0.0 to 1.0)")
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != "whisper-1":
            raise ValueError("Only 'whisper-1' model is supported")
        return v
    
    @field_validator('response_format')
    @classmethod
    def validate_format(cls, v):
        if v is None:
            return 'json'
        if v not in ['json', 'text', 'srt', 'verbose_json', 'vtt']:
            raise ValueError(f"Unsupported format: {v}")
        return v
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is None:
            return 0.0
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0")
        return v

class TaskResult(BaseModel):
    text: str
    language: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    task: Optional[str] = None
    duration: Optional[float] = None

# Global state
config = AppConfig()
logger = setup_logging(config.LOG_DIR, config.DEBUG)
security = HTTPBearer(auto_error=False)

# Use limiter with Redis backend if available
try:
    redis_client = Redis(
        host=config.REDIS_HOST, 
        port=config.REDIS_PORT, 
        db=config.REDIS_DB, 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()  # Test connection
    logger.info(f"Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
    limiter = Limiter(key_func=get_remote_address, storage_uri=f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}")
except Exception as e:
    logger.warning(f"Redis connection failed: {str(e)}. Using in-memory storage")
    limiter = Limiter(key_func=get_remote_address)

# Thread pools
worker_pool = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_TASKS)

# Task and cache management
active_tasks: Dict[str, Dict[str, Any]] = {}
transcription_cache: Dict[str, Dict[str, Any]] = {}
task_last_accessed: Dict[str, float] = {}
model_last_used = time.time()

# Whisper model and settings
whisper_model = None
whisper_model_lock = asyncio.Lock()
random = __import__('random')

WHISPER_DEFAULT_SETTINGS = {
    "temperature": 0.0,
    "no_speech_threshold": 0.25,  # More sensitive speech detection
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "task": "transcribe",
    "beam_size": 1,  # Reduced beam size for faster inference
    "best_of": 1,    # Only keep best result
    "max_initial_timestamp": 1.0,  # Fast initial timestamp detection
    "vad_filter": True,  # Enable voice activity detection
    "vad_threshold": 0.25  # Lower VAD threshold for better sensitivity
}

# Performance metrics
performance_metrics = {
    "total_requests": 0,
    "success_count": 0,
    "error_count": 0,
    "average_response_time": 0,
    "response_times": [],
    "peak_memory_usage": 0,
    "current_memory_usage": 0,
    "model_load_count": 0
}

# Utility functions
def update_metrics(status_code: int, duration: float):
    """Update performance metrics"""
    performance_metrics["total_requests"] += 1
    performance_metrics["response_times"].append(duration)
    
    # Limit stored response times to avoid memory growth
    if len(performance_metrics["response_times"]) > 1000:
        performance_metrics["response_times"] = performance_metrics["response_times"][-1000:]
    
    performance_metrics["average_response_time"] = sum(performance_metrics["response_times"]) / len(performance_metrics["response_times"])
    
    if 200 <= status_code < 400:
        performance_metrics["success_count"] += 1
    else:
        performance_metrics["error_count"] += 1
    
    # Update memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    performance_metrics["current_memory_usage"] = memory_usage_mb
    
    if memory_usage_mb > performance_metrics["peak_memory_usage"]:
        performance_metrics["peak_memory_usage"] = memory_usage_mb

def log_system_metrics():
    """Log system resource usage metrics"""
    # Memory usage (RAM)
    memory = psutil.virtual_memory()
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    # Disk usage
    disk = psutil.disk_usage('/')
    
    logger.info(
        f"System metrics: CPU={cpu_percent}%, "
        f"Memory={memory.percent}% ({memory.used/1024/1024:.1f}MB/{memory.total/1024/1024:.1f}MB), "
        f"Disk={disk.percent}% ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)"
    )
    
    # Check if memory usage is approaching threshold
    if memory.percent > config.MEMORY_THRESHOLD * 100:
        logger.warning(f"Memory usage above threshold: {memory.percent}%")
        # Force garbage collection
        gc.collect()
        if 'whisper_model' in globals() and whisper_model is not None:
            logger.warning("High memory pressure - consider unloading model")

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_audio_file(file_path: str) -> bool:
    """Validate audio file including silence check"""
    try:
        audio = decode_audio(file_path)
        duration = len(audio) / 16000
        
        logger.debug(f"Audio validation: {duration:.2f}s length, {len(audio)} samples")
        peaks = np.abs(audio).max()
        logger.debug(f"Audio peaks: {peaks:.4f} (max possible=1.0)")

        if len(audio) == 0:
            logger.error("Decoded audio is empty")
            return False

        # More sophisticated audio validation
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
        logger.debug(f"Audio RMS: {rms:.4f}")
        
        # Check if there's any significant audio content
        frame_length = 1024
        hop_length = 512
        frames = np.array([audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)])
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        
        # If we have any frames with significant audio content
        has_speech = np.any(frame_rms > 0.0005)  # More lenient threshold
        if not has_speech:
            logger.warning("No significant audio content detected")
            # Don't reject yet, let Whisper try to process it
        
        if duration > config.MAX_AUDIO_DURATION:
            logger.warning(f"Audio duration exceeds limit: {duration}s")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

async def load_whisper_model():
    """Load whisper model with memory optimizations"""
    global whisper_model, model_last_used
    
    async with whisper_model_lock:
        if whisper_model is not None:
            # Model already loaded
            model_last_used = time.time()
            return whisper_model
            
        logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
        
        try:
            # Optimize CPU performance
            torch.set_num_threads(min(4, os.cpu_count() or 4))
            torch.set_num_interop_threads(1)  # Reduce inter-op parallelism
            torch.backends.quantized.engine = 'qnnpack'
            
            # Load model with optimized settings
            whisper_model = WhisperModel(
                config.WHISPER_MODEL,
                device="cpu",
                compute_type="int8",  # Use int8 quantization
                cpu_threads=2,        # Limit CPU threads
                num_workers=1,        # Reduce worker threads
                download_root=config.CACHE_DIR,
                local_files_only=True  # Avoid downloads during runtime
            )
            
            # Record metrics
            performance_metrics["model_load_count"] += 1
            model_last_used = time.time()
            
            logger.info("Whisper model loaded successfully")
            return whisper_model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
        finally:
            if whisper_model is None:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

async def unload_whisper_model_if_idle():
    """Unload whisper model if it's been idle for too long"""
    global whisper_model
    
    if whisper_model is None:
        return
        
    idle_minutes = (time.time() - model_last_used) / 60
    
    if idle_minutes >= config.MODEL_UNLOAD_IDLE_MINUTES:
        async with whisper_model_lock:
            if whisper_model is not None:
                logger.info(f"Unloading model after {idle_minutes:.1f} minutes of inactivity")
                del whisper_model
                whisper_model = None
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

async def periodic_cleanup():
    """Periodic cleanup task for managing memory and cache"""
    while True:
        try:
            # Clean up old tasks
            current_time = time.time()
            expired_tasks = []
            
            for task_id, last_accessed in task_last_accessed.items():
                if current_time - last_accessed > config.TASK_RETENTION_HOURS * 3600:
                    expired_tasks.append(task_id)
            
            for task_id in expired_tasks:
                if task_id in active_tasks:
                    del active_tasks[task_id]
                if task_id in task_last_accessed:
                    del task_last_accessed[task_id]
                if task_id in webhooks:
                    del webhooks[task_id]
                    
            logger.debug(f"Cleaned up {len(expired_tasks)} expired tasks")
            
            # Check if model should be unloaded
            await unload_whisper_model_if_idle()
            
            # Log current system metrics
            log_system_metrics()
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)
            
        await asyncio.sleep(300)  # Run every 5 minutes

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key provided in the request"""
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
    
    if not hmac.compare_digest(credentials.credentials, config.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials

# Transcription core functionality
async def async_transcribe(audio_path: str, task_id: str, **whisper_args):
    """Async wrapper for transcription"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        worker_pool, 
        lambda: transcribe(audio_path, task_id, **whisper_args)
    )

def transcribe(audio_path: str, task_id: str, **whisper_args):
    """Transcribe audio with optimized chunking for memory-constrained environments"""
    global model_last_used, whisper_model

    # Update task status
    active_tasks[task_id]["status"] = TaskStatus.PROCESSING
    
    # Handle temperature settings safely
    temperature_increment = whisper_args.pop('temperature_increment_on_fallback', None)
    base_temperature = whisper_args.get('temperature', 0.0)
    
    if temperature_increment is not None:
        whisper_args['temperature'] = [
            base_temperature + i * temperature_increment
            for i in range(int((1.0 - base_temperature) / temperature_increment) + 1)
        ]
    else:
        whisper_args['temperature'] = [base_temperature]

    logger.debug(f"Transcribing with args: {whisper_args}")
    start_time = time.time()
    
    try:
        # Load model synchronously if needed
        if whisper_model is None:
            logger.info("Loading model from worker thread")
            torch.set_num_threads(min(4, os.cpu_count() or 4))
            torch.backends.quantized.engine = 'qnnpack'
            
            whisper_model = WhisperModel(
                config.WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=config.CACHE_DIR,
                cpu_threads=min(2, os.cpu_count() or 2),
                local_files_only=False
            )
            performance_metrics["model_load_count"] += 1

        # Process audio in optimized chunks with overlap
        model_last_used = time.time()
        chunk_size = config.CHUNK_SIZE_SECONDS  # seconds
        overlap = config.CHUNK_OVERLAP  # seconds of overlap between chunks
        
        # Load audio file using faster_whisper's decode_audio
        audio = decode_audio(audio_path)
        if len(audio) == 0:
            raise ValueError("Decoded audio is empty")
        
        total_duration = len(audio) / 16000  # faster-whisper uses fixed 16kHz sample rate
        logger.info(f"Audio duration: {total_duration:.2f} seconds")
        
        transcripts = []
        all_segments = []
        
        # Process in chunks with overlap
        current_pos = 0
        while current_pos < int(total_duration):
            end_pos = min(current_pos + chunk_size, total_duration)
            
            # Calculate chunk boundaries with overlap
            start_sample = max(0, int((current_pos - overlap) * 16000))
            end_sample = min(len(audio), int((end_pos + overlap) * 16000))
            
            logger.debug(f"Processing chunk {current_pos}-{end_pos} seconds")
            
            # Extract audio chunk
            audio_chunk = audio[start_sample:end_sample]
            
            # Transcribe chunk
            supported_args = {
                'language': whisper_args.get('language'),
                'task': whisper_args.get('task', 'transcribe'),
                'temperature': whisper_args.get('temperature', [0.0])[0],
                'no_speech_threshold': whisper_args.get('no_speech_threshold', 0.3),
                'compression_ratio_threshold': whisper_args.get('compression_ratio_threshold', 2.4),
                'condition_on_previous_text': whisper_args.get('condition_on_previous_text', True),
                'beam_size': whisper_args.get('beam_size', 5)
            }
            
            # Remove None values
            supported_args = {k: v for k, v in supported_args.items() if v is not None}
            
            # Transcribe chunk
            transcription_result = whisper_model.transcribe(
                audio_chunk,
                **supported_args
            )
            
            # Handle transcription result
            if transcription_result and len(transcription_result) == 2:
                segments, info = transcription_result
                
                # Convert segments to expected format and adjust timestamps
                if segments:
                    for segment in segments:
                        # Adjust segment timestamps to account for chunk position and overlap
                        segment_start = segment.start + (current_pos - overlap)
                        segment_end = segment.end + (current_pos - overlap)
                        
                        # Only include segments that fall within the current chunk (excluding overlap)
                        if segment_end > current_pos and segment_start < end_pos:
                            all_segments.append({
                                "start": max(0, segment_start),
                                "end": min(total_duration, segment_end),
                                "text": segment.text.strip()
                            })
                    
                    # Get transcript text from valid segments
                    chunk_text = ' '.join(segment.text.strip() for segment in segments 
                                        if segment.end + (current_pos - overlap) > current_pos 
                                        and segment.start + (current_pos - overlap) < end_pos)
                    if chunk_text:
                        transcripts.append(chunk_text)

            # Clear memory after each chunk
            del audio_chunk
            if len(transcripts) % config.GC_FREQUENCY == 0:  # Run GC every N chunks
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Move to next chunk
            current_pos = end_pos
        
        # Combine results and remove duplicates
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Remove duplicate or overlapping segments
        if all_segments:
            filtered_segments = [all_segments[0]]
            for segment in all_segments[1:]:
                prev = filtered_segments[-1]
                # Check if segments overlap significantly
                if segment["start"] > prev["end"] - 0.5:  # Allow small overlap
                    filtered_segments.append(segment)
                elif segment["text"] != prev["text"]:  # Different text, might be a correction
                    if len(segment["text"]) > len(prev["text"]):  # Keep the longer text
                        filtered_segments[-1] = segment
        else:
            filtered_segments = []
        
        # Combine results
        combined_text = ' '.join(s["text"] for s in filtered_segments).strip()
        if not combined_text:
            logger.warning("Transcription resulted in empty text")
        
        combined_transcript = {
            'text': combined_text,
            'segments': filtered_segments,
            'language': whisper_args.get('language'),
            'task': whisper_args.get('task', 'transcribe'),
        }
        
        # Add duration
        combined_transcript['duration'] = total_duration if not filtered_segments else filtered_segments[-1]['end']
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        # Update task status and result
        active_tasks[task_id]["status"] = TaskStatus.COMPLETE
        active_tasks[task_id]["result"] = combined_transcript
        task_last_accessed[task_id] = time.time()
        
        # Post-process for compatibility
        if combined_transcript.get('language') == 'ja':
            combined_transcript['language'] = 'japanese'
            
        return combined_transcript
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        active_tasks[task_id]["status"] = TaskStatus.FAILED
        active_tasks[task_id]["error"] = str(e)
        raise

def generate_task_id() -> str:
    """Generate unique task ID"""
    return str(uuid.uuid4())

def format_response(result: dict, response_format: str) -> Union[dict, str]:
    """Format transcription result with silent audio handling"""
    # Base response with fallbacks
    base = {
        'text': result.get('text', '').strip() or "No speech detected",
        'language': result.get('language'),
        'segments': result.get('segments', []),
        'duration': result.get('duration', 0.0),
        'task': result.get('task', 'transcribe')
    }

    if response_format == 'text':
        return base['text']
        
    elif response_format == 'srt':
        ret = ""
        for i, seg in enumerate(base['segments'], 1):
            td_s = timedelta(seconds=seg.get("start", 0))
            td_e = timedelta(seconds=seg.get("end", 0))
            
            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02},{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02},{td_e.microseconds//1000:03}'
            
            ret += f'{i}\n{t_s} --> {t_e}\n{seg.get("text", "[silence]")}\n\n'
        return ret.strip() or "1\n00:00:00,000 --> 00:00:00,000\n[silence]"
        
    elif response_format == 'vtt':
        ret = "WEBVTT\n\n"
        for seg in base['segments']:
            td_s = timedelta(seconds=seg.get("start", 0))
            td_e = timedelta(seconds=seg.get("end", 0))
            
            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}.{td_e.microseconds//1000:03}'
            
            ret += f"{t_s} --> {t_e}\n{seg.get('text', '[silence]')}\n\n"
        return ret.strip() or "WEBVTT\n\n00:00:00.000 --> 00:00:00.000\n[silence]"
        
    elif response_format == 'verbose_json':
        return {**base, 'segments': [
            {**seg, 'text': seg.get('text', '[silence]')} 
            for seg in base['segments']
        ]}
        
    else:  # json (default)
        return {
            'text': base['text'],
            'language': base['language'],
            'segments': base['segments'],
            'duration': base['duration']
        }

# Webhook handling
webhooks = {}  # {task_id: {"url": webhook_url, "secret": webhook_secret}}

def send_webhook_notification(task_id: str):
    """Send webhook notification for task completion"""
    if task_id not in webhooks:
        return
    
    webhook = webhooks[task_id]
    task = active_tasks[task_id]
    
    payload = {
        "task_id": task_id,
        "status": task["status"],
    }
    
    if task["status"] == TaskStatus.COMPLETE:
        payload["result"] = {"text": task["result"].get('text', '')}
    elif task["status"] == TaskStatus.FAILED:
        payload["error"] = task.get("error", "Unknown error")
    
    headers = {"Content-Type": "application/json"}
    if webhook['secret']:
        signature = hmac.new(
            webhook['secret'].encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        headers['X-Webhook-Signature'] = signature
    
    try:
        response = requests.post(
            webhook['url'],
            json=payload,
            headers=headers,
            timeout=5
        )
        logger.info(f"Webhook delivery for task {task_id}: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook delivery failed for task {task_id}: {str(e)}")

# FastAPI App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown operations"""
    # Start background task for periodic cleanup
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    logger.info(f"Starting Whisper API server with model: {config.WHISPER_MODEL}")
    logger.info(f"Server configured to use up to {config.MAX_CONCURRENT_TASKS} concurrent tasks")
    
    yield
    
    # Shutdown operations
    logger.info("Shutting down, cleaning up resources")
    cleanup_task.cancel()
    
    # Release model resources
    global whisper_model
    if whisper_model:
        logger.info("Unloading Whisper model")
        del whisper_model
        whisper_model = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Shutdown thread pool
    worker_pool.shutdown(wait=False)

app = FastAPI(
    title="Whisper API",
    description="API for audio transcription using Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request duration and add it to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded", 
            "retry_after": getattr(exc, 'retry_after', 60)
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred", "detail": str(exc)},
    )

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint with system metrics"""
    memory = psutil.virtual_memory()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "time": time.time(),
        "metrics": {
            "requests": performance_metrics["total_requests"],
            "success_rate": (performance_metrics["success_count"] / max(1, performance_metrics["total_requests"])) * 100,
            "avg_response_time": performance_metrics["average_response_time"],
            "system": {
                "memory_used_percent": memory.percent,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
            }
        }
    }

@app.get("/v1/audio/transcriptions/{task_id}")
async def get_transcription_status(
    task_id: str, 
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    api_key: str = Depends(verify_api_key)
):
    """
    Get the status or result of a transcription task with pagination.
    
    Returns either the completion status or the full transcription results
    with pagination for large transcriptions.
    """
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID not found: {task_id}"
        )

    # Update last accessed time for this task
    task_last_accessed[task_id] = time.time()
    task = active_tasks[task_id]
    
    if task["status"] != TaskStatus.COMPLETE:
        return {"status": task["status"]}
    
    # For completed transcriptions, paginate the results
    result = task["result"]
    total_segments = len(result.get('segments', []))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_segments = result.get('segments', [])[start_idx:end_idx]
    has_more = end_idx < total_segments
    
    return {
        "status": "complete",
        "text": result.get('text', ''),
        "language": result.get('language'),
        "total_segments": total_segments,
        "page": page,
        "page_size": page_size,
        "has_more": has_more,
        "segments": paginated_segments
    }

@app.post('/v1/webhooks')
@limiter.limit(config.RATE_LIMITS[1])  # Apply less strict rate limit for webhooks
async def register_webhook(
    request: Request,
    webhook: Webhook,
    task_id: str = Query(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Register a webhook for transcription completion notifications.
    
    The webhook will be called when the transcription task completes or fails.
    """
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID not found: {task_id}"
        )
    
    webhooks[task_id] = {
        "url": str(webhook.url),
        "secret": webhook.secret
    }
    
    return {"message": "Webhook registered successfully", "task_id": task_id}

@app.post('/v1/audio/transcriptions')
@limiter.limit(config.RATE_LIMITS[0])
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    """
    Transcribe audio file to text.
    
    This endpoint accepts audio files and transcribes them using the Whisper model.
    It supports both synchronous and asynchronous (webhook-based) transcription.
    """
    # Process and validate request parameters
    try:
        req = TranscriptionRequest(
            model=model,
            response_format=response_format,
            language=language,
            prompt=prompt,
            temperature=temperature
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Check current system load
    memory = psutil.virtual_memory()
    if memory.percent > 85:  # Over 85% memory usage
        logger.warning(f"System under high memory load: {memory.percent}%")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server under high load, please try again later",
            headers={"Retry-After": "60"}
        )
    
    # Check file size
    if hasattr(file, 'size'):
        if file.size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {config.MAX_FILE_SIZE // (1024 * 1024)} MB"
            )
    
    # Generate task ID and initialize task
    task_id = generate_task_id()
    active_tasks[task_id] = {"status": TaskStatus.PENDING, "result": None}
    task_last_accessed[task_id] = time.time()
    
    # Process webhook registration if provided
    if webhook_url:
        webhooks[task_id] = {
            "url": webhook_url,
            "secret": webhook_secret
        }
    
    logger.info(f"Processing transcription request for file: {file.filename} (Task ID: {task_id})")
    
    # Create temporary file for processing
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Save uploaded file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Validate file using faster-whisper's decoder
        if not validate_audio_file(temp_file_path):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid audio file or exceeds maximum duration"
            )
        
        # Calculate file hash for caching
        file_hash = calculate_file_hash(temp_file_path)
        
        # Check if transcription is already in cache
        if file_hash in transcription_cache:
            logger.info(f"Using cached transcription for file hash: {file_hash}")
            result = transcription_cache[file_hash]
            active_tasks[task_id]["status"] = TaskStatus.COMPLETE
            active_tasks[task_id]["result"] = result
            
            # Format and return response
            formatted_response = format_response(result, req.response_format)
            
            # If response is a string, return as PlainText for proper content-type
            if isinstance(formatted_response, str):
                return Response(content=formatted_response, media_type="text/plain")
            return formatted_response
        
        # Apply transcription settings
        settings = WHISPER_DEFAULT_SETTINGS.copy()  # Start with defaults
        
        # Override with request parameters
        if req.temperature is not None:
            settings['temperature'] = req.temperature
        if req.language is not None:
            settings['language'] = req.language
        if req.prompt is not None:
            settings['initial_prompt'] = req.prompt
            
        # Remove any incompatible settings
        settings = {k: v for k, v in settings.items() if k in [
            'temperature', 'temperature_increment_on_fallback',
            'no_speech_threshold', 'compression_ratio_threshold',
            'condition_on_previous_text', 'beam_size', 'language',
            'initial_prompt', 'task'
        ]}

        # For async requests with webhook, process in background
        if webhook_url:
            # Create task for background processing
            async def process_in_background():
                try:
                    # Ensure model is loaded
                    await load_whisper_model()
                    
                    # Process transcription
                    result = await async_transcribe(temp_file_path, task_id, **settings)
                    
                    # Cache result
                    transcription_cache[file_hash] = result
                    
                    # Send webhook notification
                    send_webhook_notification(task_id)
                except Exception as e:
                    logger.error(f"Background transcription failed: {str(e)}", exc_info=True)
                    active_tasks[task_id]["status"] = TaskStatus.FAILED
                    active_tasks[task_id]["error"] = str(e)
                    
                    # Send failure notification
                    send_webhook_notification(task_id)
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.error("Failed to remove temporary file")
            
            # Start background task
            asyncio.create_task(process_in_background())
            
            return {
                "task_id": task_id,
                "status": "processing",
                "message": "Transcription started - check status or await webhook notification"
            }
        
        # For synchronous requests, process immediately
        await load_whisper_model()
        result = await async_transcribe(temp_file_path, task_id, **settings)
        
        # Cache result for future use
        transcription_cache[file_hash] = result
        
        # Format and return response
        formatted_response = format_response(result, req.response_format)
        
        # If response is a string, return as PlainText for proper content-type
        if isinstance(formatted_response, str):
            return Response(content=formatted_response, media_type="text/plain")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Transcription request failed: {str(e)}", exc_info=True)
        
        # Update task status if it was a background task
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = TaskStatus.FAILED
            active_tasks[task_id]["error"] = str(e)
            
            # Send webhook notification if registered
            if task_id in webhooks:
                send_webhook_notification(task_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug("Removed temporary file")
            except Exception as e:
                logger.error("Failed to remove temporary file")

def main():
    """Entry point for running the server directly"""
    # Validate configuration
    try:
        # Set up metrics logging thread
        log_system_metrics()
        
        # Initialize libraries
        mimetypes.init()
        
        # Start the server
        uvicorn.run(
            "server:app", 
            host=config.HOST, 
            port=config.PORT,
            log_level="debug" if config.DEBUG else "info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
