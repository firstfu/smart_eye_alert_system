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

    # 串流設定
    STREAM_FPS: int = 20  # 串流影像幀率
    STREAM_QUALITY: int = 80  # JPEG 壓縮品質 (0-100)
    MAX_CONNECTIONS_PER_CAMERA: int = 10  # 每個攝影機的最大連接數
    STREAM_BUFFER_SIZE: int = 5  # 串流緩衝區大小

    # 影像品質設定
    MIN_BRIGHTNESS: float = 50.0  # 最小亮度值 (0-255)
    MIN_CONTRAST: float = 30.0    # 最小對比度值
    MIN_BLUR_SCORE: float = 100.0 # 最小清晰度分數
    QUALITY_CHECK_INTERVAL: float = 5.0  # 品質檢查間隔（秒）
    MIN_QUALITY_SCORE: float = 60.0  # 最低品質分數

    # 移動偵測設定
    MOTION_HISTORY: int = 500  # 背景模型歷史長度
    MOTION_THRESHOLD: float = 16.0  # 移動偵測閾值
    MOTION_RATIO_THRESHOLD: float = 0.01  # 最小移動區域比例

    # 場景變化設定
    SCENE_CHANGE_THRESHOLD: float = 0.15  # 場景變化閾值

    # 事件處理設定
    MAX_EVENTS_HISTORY: int = 1000  # 最大事件歷史記錄數
    EVENT_CLEANUP_INTERVAL: float = 3600.0  # 事件清理間隔（秒）

    # 影像增強設定
    AUTO_ENHANCE: bool = True  # 是否自動增強影像
    CLAHE_CLIP_LIMIT: float = 3.0  # CLAHE 限制值
    SHARPNESS_FACTOR: float = 1.5  # 銳化強度

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