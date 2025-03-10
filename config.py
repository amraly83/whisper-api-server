from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Existing settings
    api_key: str = "your-secure-key"
    max_duration: int = 600
    rate_limit: int = 100
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
