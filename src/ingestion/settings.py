from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

class Settings(BaseSettings):
    entsoe_api_key: str
    
    model_config = SettingsConfigDict(
        env_file=(
            BASE_DIR / ".env",
            SRC_DIR / ".env",
            PROJECT_ROOT / ".env",
        ),
        extra="ignore"
    )