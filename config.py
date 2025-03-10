from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Existing settings...
    api_key: str
    max_duration: int = 600
    rate_limit: int = 100
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list = ["*"]
    cache_ttl: int = 3600
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
