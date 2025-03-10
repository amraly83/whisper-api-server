from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    whisper_model_size: str = "base"
    compute_type: str = "int8"
    device: str = "cpu"
    beam_size: int = 3
    max_workers: int = 4  # Number of CPU cores
    
    model_config = SettingsConfigDict(extra="ignore")

settings = Settings()
