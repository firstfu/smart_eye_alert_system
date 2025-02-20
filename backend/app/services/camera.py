import cv2
import numpy as np
from typing import Optional, Generator
from ..core.config import settings

class Camera:
    def __init__(self, camera_id: str = "0"):
        """初始化攝影機

        Args:
            camera_id: 攝影機ID，可以是數字（本地攝影機）或 RTSP URL
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_rtsp = isinstance(camera_id, str) and camera_id.startswith("rtsp://")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """啟動攝影機串流"""
        if self.is_rtsp:
            self.cap = cv2.VideoCapture(self.camera_id)
        else:
            self.cap = cv2.VideoCapture(int(self.camera_id))

        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 {self.camera_id}")

        # 設定影像大小
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    def stop(self):
        """停止攝影機串流"""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """獲取單一影像幀

        Returns:
            numpy.ndarray: 影像幀數據，如果讀取失敗則返回 None
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def generate_frames(self) -> Generator[np.ndarray, None, None]:
        """生成影像幀串流

        Yields:
            numpy.ndarray: 影像幀數據
        """
        while True:
            frame = self.get_frame()
            if frame is None:
                break

            # 進行基本的影像處理
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            yield frame