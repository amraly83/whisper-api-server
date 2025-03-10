from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    device: str = "cuda"
    whisper_model_size: str = "base"
    api_key: str
    max_duration: int = 600
    rate_limit: int = 100
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list = ["*"]
    cache_ttl: int = 3600
    # Maximum number of worker threads for audio processing. Defaults to 1.
    max_workers: int = 1
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
