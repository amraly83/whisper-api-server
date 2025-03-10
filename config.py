from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    whisper_model_size: str = Field(default="small", env="MODEL_SIZE")
    compute_type: str = Field(default="int8", env="COMPUTE_TYPE")
    device: str = Field(default="cpu", env="DEVICE")
    beam_size: int = Field(default=3, env="BEAM_SIZE")

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",          # Load variables from .env file
        env_file_encoding="utf-8"
    )

settings = Settings()
