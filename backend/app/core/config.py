from pydantic_settings import BaseSettings
from typing import Optional, List
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

    # 物件偵測設定
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"  # YOLO 模型路徑
    DETECTION_INTERVAL: float = 0.5  # 偵測間隔（秒）
    DETECTION_CONFIDENCE: float = 0.25  # 偵測信心度閾值
    MAX_DETECTIONS: int = 50  # 每幀最大偵測數量
    ENABLE_GPU: bool = True  # 是否啟用 GPU
    DRAW_DETECTIONS: bool = True  # 是否在串流中繪製偵測結果
    DETECTION_CLASSES: List[str] = []  # 要偵測的類別（空列表表示全部類別）

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

    # LLM 設定
    LLM_MODEL = "gpt-4"
    OPENAI_API_KEY = ""  # 需要從環境變數讀取
    OPENAI_API_BASE = "https://api.openai.com/v1"
    LLM_CACHE_SIZE = 1000  # 快取大小
    LLM_CACHE_TTL = 300   # 快取存活時間（秒）
    LLM_RATE_LIMIT = 20   # 每分鐘最大請求數

    # 場景分析設定
    SCENE_ANALYSIS_INTERVAL = 10  # 場景分析間隔（秒）
    MIN_CONFIDENCE_THRESHOLD = 0.6  # 最小信心度閾值
    RISK_ALERT_THRESHOLD = 0.8     # 風險警報閾值

    # 行為分析設定
    BEHAVIOR_TRACKING_ENABLED: bool = True
    MAX_TRAJECTORY_LENGTH: int = 50
    STATIONARY_THRESHOLD: float = 0.1
    INTERACTION_DISTANCE: float = 100.0
    FALLEN_ANGLE_THRESHOLD: float = 45.0
    RAPID_MOVEMENT_THRESHOLD: float = 5.0
    LONG_STATIONARY_THRESHOLD: float = 300.0  # 秒
    INTERACTION_TIMEOUT: float = 5.0  # 秒

    # 姿態估計設定
    POSE_DETECTION_CONFIDENCE: float = 0.5
    POSE_TRACKING_CONFIDENCE: float = 0.5
    POSE_MODEL_COMPLEXITY: int = 1
    POSE_STATIC_IMAGE_MODE: bool = False
    POSE_SMOOTH_LANDMARKS: bool = True

    # 跌倒偵測參數
    FALL_BODY_TILT_THRESHOLD: float = 60.0  # 身體傾斜角度閾值
    FALL_HEIGHT_RATIO_THRESHOLD: float = 0.5  # 身體高度比例閾值
    FALL_STABILITY_THRESHOLD: float = 0.3  # 穩定性分數閾值
    FALL_DETECTION_INTERVAL: float = 0.1  # 跌倒偵測間隔（秒）
    FALL_CONFIRMATION_FRAMES: int = 3  # 確認跌倒所需的連續幀數

    # 執行緒池設定
    CPU_CORES: int = os.cpu_count() or 4  # CPU 核心數
    MAX_WORKERS: int = CPU_CORES * 2      # 最大工作執行緒數
    TASK_QUEUE_SIZE: int = 1000           # 任務佇列大小
    TASK_TIMEOUT: float = 30.0            # 任務超時時間（秒）

    # 任務優先級設定
    PRIORITY_HIGH: int = 1    # 高優先級（如跌倒偵測）
    PRIORITY_NORMAL: int = 2  # 一般優先級（如物件偵測）
    PRIORITY_LOW: int = 3     # 低優先級（如場景分析）

    # 效能監控設定
    STATS_UPDATE_INTERVAL: float = 5.0  # 統計資訊更新間隔（秒）
    PERFORMANCE_LOG_INTERVAL: float = 60.0  # 效能日誌記錄間隔（秒）

    # GPU 設定
    GPU_MEMORY_FRACTION: float = 0.8  # GPU 記憶體使用上限比例
    CUDA_VISIBLE_DEVICES: str = "0"  # 可見的 GPU 設備
    GPU_BATCH_SIZE: int = 4  # GPU 批次處理大小
    GPU_MEMORY_GROWTH: bool = True  # 是否允許 GPU 記憶體動態增長

    # GPU 效能監控
    GPU_STATS_INTERVAL: float = 5.0  # GPU 狀態更新間隔（秒）
    GPU_TEMPERATURE_THRESHOLD: float = 80.0  # GPU 溫度警告閾值
    GPU_MEMORY_THRESHOLD: float = 0.9  # GPU 記憶體使用警告閾值
    GPU_UTILIZATION_THRESHOLD: float = 0.95  # GPU 使用率警告閾值

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()