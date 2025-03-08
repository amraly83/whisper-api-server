import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_size: str = os.getenv("MODEL_SIZE", "tiny")
    compute_type: str = os.getenv("COMPUTE_TYPE", "int8")
    device: str = os.getenv("DEVICE", "cpu")
    
    class Config:
        env_file = ".env"

settings = Settings()