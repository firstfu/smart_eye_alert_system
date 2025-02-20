from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

class Settings(BaseSettings):
    # 基本設定
    APP_NAME: str = "智眼警示系統"
    API_V1_STR: str = "/api/v1"

    # 影像設定
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    MIN_FPS: float = 15.0

    # 影像品質設定
    MIN_BRIGHTNESS: float = 50.0  # 最小亮度值 (0-255)
    MIN_CONTRAST: float = 30.0    # 最小對比度值
    MIN_BLUR_SCORE: float = 100.0 # 最小清晰度分數
    QUALITY_CHECK_INTERVAL: float = 5.0  # 品質檢查間隔（秒）

    # 攝影機設定
    MAX_CAMERAS: int = 16  # 最大攝影機數量
    CAMERA_RECONNECT_ATTEMPTS: int = 3  # 重連嘗試次數
    CAMERA_RECONNECT_DELAY: float = 5.0  # 重連延遲（秒）

    # 快取設定
    FRAME_CACHE_SIZE: int = 30  # 每個攝影機的影像快取大小

    # 安全設定
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # 資料庫設定
    SQLITE_URL: str = "sqlite:///./smart_eye.db"

    # 通知設定
    LINE_NOTIFY_TOKEN: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()