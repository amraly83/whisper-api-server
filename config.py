from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    whisper_model_size: str = "small"
    compute_type: str = "int8"
    device: str = "cpu"
    beam_size: int = 3

    model_config: SettingsConfigDict = SettingsConfigDict(
        extra="ignore"  # Ignore unexpected environment variables
    )

settings = Settings()
