from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API 設定
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Smart Eye Alert System"

    # 資料庫設定
    SQLITE_URL: str = "sqlite:///./smart_eye.db"

    # JWT 設定
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # LINE Notify 設定
    LINE_NOTIFY_TOKEN: Optional[str] = os.getenv("LINE_NOTIFY_TOKEN")

    # 影像處理設定
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS: int = 30

    # AI 模型設定
    DETECTION_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.5

    class Config:
        case_sensitive = True

settings = Settings()