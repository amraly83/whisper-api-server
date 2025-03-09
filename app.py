from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
from config import settings

app = FastAPI(title="Whisper API Server")

# Fix the model initialization - the first parameter is positional, not named
try:
    print(f"Loading model: size={settings.whisper_model_size}, device={settings.device}, compute_type={settings.compute_type}")
    # The model_size should be a positional argument, not a keyword argument
    model_name = WhisperModel(
        settings.whisper_model_size,  # Remove the parameter name
        device=settings.device, 
        compute_type=settings.compute_type
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.get("/")
async def root():
    return {
        "message": "Whisper API is running",
        "model_size": settings.whisper_model_size,
        "device": settings.device
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str = Form(default=None),
    prompt: str = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0),
):

    # Validate response format
    if response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(status_code=400, detail="Unsupported response format")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    print(f"Created temporary file: {temp_file.name}")
    
    try:
        # Save the uploaded file to the temporary file
        with open(temp_file.name, "wb") as buffer:
            buffer.write(await file.read())
        
        # Get file extension and check if it's supported
        file_ext = os.path.splitext(file.filename)[1].lower()
        supported_formats = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".oga"]
        
        if file_ext not in supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")
        
        # Transcribe the audio
        print(f"Starting transcription for file: {file.filename}")
        segments, info = model.transcribe(
            temp_file.name,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=5,
        )
        print("Transcription completed successfully")
        
        # Prepare the response
        text = ""
        segments_data = []
        
        for segment in segments:
            text += segment.text
            segments_data.append({
                "id": len(segments_data),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": segment.avg_logprob,
            })
        
        # Format response based on requested format
        if response_format == "text":
            return text
        elif response_format == "json":
            return {"text": text}
        elif response_format == "verbose_json":
            return {
                "text": text,
                "segments": segments_data,
                "language": info.language,
            }
        elif response_format == "srt" or response_format == "vtt":
            # This is a simplification - proper SRT/VTT formatting would require more work
            subtitle_content = ""
            for i, segment in enumerate(segments_data):
                subtitle_content += f"{i+1}\n"
                start_time = format_timestamp(segment["start"], response_format)
                end_time = format_timestamp(segment["end"], response_format)
                subtitle_content += f"{start_time} --> {end_time}\n"
                subtitle_content += f"{segment['text']}\n\n"
            return subtitle_content
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full traceback
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)
        print(f"Deleted temporary file: {temp_file.name}")

def format_timestamp(seconds, format_type):
    """Convert seconds to SRT or VTT timestamp format"""
    hours = int(seconds / 3600)
    minutes = int((seconds - (hours * 3600)) / 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{int((secs - int(secs)) * 1000):03d}"
    else:  # vtt
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{int((secs - int(secs)) * 1000):03d}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
