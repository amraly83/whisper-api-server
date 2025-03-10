import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    whisper_model_size: str = os.getenv("MODEL_SIZE", "small")
    compute_type: str = os.getenv("COMPUTE_TYPE","float16")  # Set to int8 for quantization
    device: str = os.getenv("DEVICE","cpu")  # Explicitly set to CPU
    beam_size: int = int(os.getenv("BEAM_SIZE",3)) # Added beam_size setting, default to 3

    model_config = {
        'protected_namespaces': ('settings_',)
    }

settings = Settings()
