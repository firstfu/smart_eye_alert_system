import cv2
import numpy as np
import time
from typing import Optional, Generator, Dict
from dataclasses import dataclass
from ..core.config import settings

@dataclass
class ImageQuality:
    brightness: float
    contrast: float
    blur_score: float
    timestamp: float

class Camera:
    def __init__(self, camera_id: str = "0"):
        """初始化攝影機

        Args:
            camera_id: 攝影機ID，可以是數字（本地攝影機）或 RTSP URL
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_rtsp = isinstance(camera_id, str) and camera_id.startswith("rtsp://")
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.quality_history: Dict[float, ImageQuality] = {}
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.is_connected = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> bool:
        """啟動攝影機串流

        Returns:
            bool: 是否成功啟動
        """
        try:
            if self.is_rtsp:
                self.cap = cv2.VideoCapture(self.camera_id)
            else:
                self.cap = cv2.VideoCapture(int(self.camera_id))

            if not self.cap.isOpened():
                raise RuntimeError(f"無法開啟攝影機 {self.camera_id}")

            # 設定影像大小
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

            self.is_connected = True
            self.reconnect_attempts = 0
            return True

        except Exception as e:
            self.is_connected = False
            return False

    def stop(self):
        """停止攝影機串流"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False

    def try_reconnect(self) -> bool:
        """嘗試重新連接攝影機

        Returns:
            bool: 是否成功重連
        """
        current_time = time.time()

        # 檢查是否需要等待重連延遲
        if current_time - self.last_reconnect_time < settings.CAMERA_RECONNECT_DELAY:
            return False

        # 檢查重連次數是否超過限制
        if self.reconnect_attempts >= settings.CAMERA_RECONNECT_ATTEMPTS:
            return False

        self.stop()  # 確保先停止現有連接
        self.reconnect_attempts += 1
        self.last_reconnect_time = current_time

        if self.start():
            self.is_connected = True
            self.reconnect_attempts = 0
            return True

        return False

    def assess_image_quality(self, frame: np.ndarray) -> ImageQuality:
        """評估影像品質

        Args:
            frame: 影像幀數據

        Returns:
            ImageQuality: 影像品質評估結果
        """
        # 計算亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # 計算對比度
        contrast = np.std(gray)

        # 計算模糊度分數（使用Laplacian算子）
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        quality = ImageQuality(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            timestamp=time.time()
        )

        # 儲存品質歷史記錄
        self.quality_history[quality.timestamp] = quality

        # 只保留最近 100 筆記錄
        if len(self.quality_history) > 100:
            oldest_key = min(self.quality_history.keys())
            del self.quality_history[oldest_key]

        return quality

    def update_fps(self):
        """更新 FPS 計算"""
        current_time = time.time()
        self.frame_count += 1

        # 每秒更新一次 FPS
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = current_time

    def get_frame(self) -> Optional[np.ndarray]:
        """獲取單一影像幀

        Returns:
            numpy.ndarray: 影像幀數據，如果讀取失敗則返回 None
        """
        if not self.is_connected:
            if not self.try_reconnect():
                return None

        if not self.cap or not self.cap.isOpened():
            self.is_connected = False
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.is_connected = False
            return None

        self.update_fps()
        return frame

    def generate_frames(self) -> Generator[tuple[np.ndarray, ImageQuality], None, None]:
        """生成影像幀串流

        Yields:
            tuple[numpy.ndarray, ImageQuality]: 影像幀數據和品質評估結果
        """
        while True:
            frame = self.get_frame()
            if frame is None:
                if not self.try_reconnect():
                    break
                continue

            # 進行基本的影像處理
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            # 評估影像品質
            quality = self.assess_image_quality(frame)

            yield frame, quality

    def get_quality_stats(self) -> Dict[str, float]:
        """獲取影像品質統計資料

        Returns:
            Dict[str, float]: 品質統計資料
        """
        if not self.quality_history:
            return {
                "avg_brightness": 0,
                "avg_contrast": 0,
                "avg_blur_score": 0,
                "fps": self.fps
            }

        qualities = list(self.quality_history.values())
        return {
            "avg_brightness": sum(q.brightness for q in qualities) / len(qualities),
            "avg_contrast": sum(q.contrast for q in qualities) / len(qualities),
            "avg_blur_score": sum(q.blur_score for q in qualities) / len(qualities),
            "fps": self.fps
        }

    def get_status(self) -> Dict[str, any]:
        """獲取攝影機狀態

        Returns:
            Dict[str, any]: 攝影機狀態資訊
        """
        return {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "reconnect_attempts": self.reconnect_attempts,
            "fps": self.fps,
            "quality_stats": self.get_quality_stats()
        }