from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, Header, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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

# Configuration
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'tiny')
API_KEY = os.environ.get('API_KEY', secrets.token_hex(16))  # Generate a random API key if not provided
PORT = int(os.environ.get('PORT', 8000))
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

# Global model
whisper_model = None

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
    # Load the ML model
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device, in_memory=True)
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

# Whisper transcription function
def transcribe(audio_path: str, **whisper_args):
    """Transcribe the audio file using whisper"""
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
    
    transcript = whisper_model.transcribe(
        audio_path,
        **whisper_args,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

    return transcript

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
        transcript = transcribe(audio_path=temp_file_path, **settings)
        
        # Format response based on requested format
        if response_format == 'text':
            return transcript['text']
            
        elif response_format == 'srt':
            ret = ""
            for seg in transcript['segments']:
                td_s = timedelta(milliseconds=seg["start"]*1000)
                td_e = timedelta(milliseconds=seg["end"]*1000)
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"])
            ret += '\n'
            return ret
            
        elif response_format == 'vtt':
            ret = "WEBVTT\n\n"
            for seg in transcript['segments']:
                td_s = timedelta(milliseconds=seg["start"]*1000)
                td_e = timedelta(milliseconds=seg["end"]*1000)
                
                t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
                t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
                
                ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
            return ret
            
        elif response_format == 'verbose_json':
            transcript.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
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

def main():
    # Print the API key to console
    logger.info(f"API Key: {API_KEY}")
    logger.info(f"Starting server on {HOST}:{PORT}")
    
    # Start the server
    uvicorn.run(
        "main:app", 
        host=HOST, 
        port=PORT, 
        log_level="debug" if DEBUG else "info"
    )

if __name__ == "__main__":
    main()