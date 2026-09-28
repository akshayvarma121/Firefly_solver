from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # Default to local frontend dev server if not specified
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost"
    ]
    
    # Defaults to 50MB
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024
    
    # Defaults to 60 seconds solve timeout
    SOLVE_TIMEOUT_SECONDS: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
