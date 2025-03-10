from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    whisper_model_size: str = "base"
    compute_type: str = "int8"
    device: str = "cpu"
    beam_size: int = 2

    model_config: SettingsConfigDict = SettingsConfigDict(
        extra="ignore"  # Ignore unexpected environment variables
    )

settings = Settings()
