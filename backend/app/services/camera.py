import cv2
import numpy as np
import time
from typing import Optional, Generator, Dict, Tuple
from dataclasses import dataclass
from ..core.config import settings
from .frame_cache import FrameCache, CachedFrame
from .gpu_manager import GPUManager
from .thread_pool import ThreadPoolManager

@dataclass
class ImageQuality:
    brightness: float
    contrast: float
    blur_score: float
    timestamp: float

    def get_overall_score(self) -> float:
        """計算整體品質分數

        Returns:
            float: 品質分數 (0-100)
        """
        # 將各項指標正規化到 0-100 範圍
        brightness_score = min(100, (self.brightness / 255.0) * 100)
        contrast_score = min(100, (self.contrast / 128.0) * 100)
        blur_score = min(100, (self.blur_score / 1000.0) * 100)

        # 加權平均
        weights = {
            "brightness": 0.3,
            "contrast": 0.3,
            "blur": 0.4
        }

        return (
            brightness_score * weights["brightness"] +
            contrast_score * weights["contrast"] +
            blur_score * weights["blur"]
        )

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
        self.frame_cache = FrameCache()

        # GPU 和多執行緒支援
        self.gpu_manager = GPUManager()
        self.thread_pool = ThreadPoolManager()
        self.device = self.gpu_manager.get_optimal_device()
        self.use_gpu = settings.ENABLE_GPU and self.device.type == 'cuda'

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

            # 如果使用 GPU，設定 CUDA 後端
            if self.use_gpu:
                self.cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_OPENCV_CUDA)

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
        self.frame_cache.clear()

        # 清理 GPU 資源
        if self.use_gpu:
            self.gpu_manager.cleanup()

        # 關閉執行緒池
        self.thread_pool.shutdown()

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
        # 提交品質評估任務到執行緒池
        task_id = self.thread_pool.submit_task(
            self._calculate_quality,
            frame,
            priority=settings.PRIORITY_LOW
        )

        try:
            quality = self.thread_pool.get_result(task_id, timeout=settings.TASK_TIMEOUT)
        except TimeoutError:
            # 如果超時，使用預設值
            quality = ImageQuality(
                brightness=0.0,
                contrast=0.0,
                blur_score=0.0,
                timestamp=time.time()
            )

        # 儲存品質歷史記錄
        self.quality_history[quality.timestamp] = quality

        # 只保留最近 100 筆記錄
        if len(self.quality_history) > 100:
            oldest_key = min(self.quality_history.keys())
            del self.quality_history[oldest_key]

        return quality

    def _calculate_quality(self, frame: np.ndarray) -> ImageQuality:
        """計算影像品質（在執行緒池中執行）

        Args:
            frame: 影像幀

        Returns:
            ImageQuality: 品質評估結果
        """
        # 如果在 GPU 上，先將影像移到 CPU
        if isinstance(frame, torch.Tensor) and frame.device.type == 'cuda':
            frame = self.gpu_manager.postprocess_from_gpu(frame)

        # 計算亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # 計算對比度
        contrast = np.std(gray)

        # 計算模糊度分數（使用Laplacian算子）
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return ImageQuality(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            timestamp=time.time()
        )

    def update_fps(self):
        """更新 FPS 計算"""
        current_time = time.time()
        self.frame_count += 1

        # 每秒更新一次 FPS
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = current_time

    def get_frame(self) -> Optional[Tuple[np.ndarray, ImageQuality]]:
        """獲取單一影像幀

        Returns:
            Optional[Tuple[np.ndarray, ImageQuality]]: (影像幀數據, 品質評估結果)，如果讀取失敗則返回 None
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

        # 如果使用 GPU，將影像移到 GPU
        if self.use_gpu:
            frame = self.gpu_manager.preprocess_for_gpu(frame)

        # 評估影像品質
        quality = self.assess_image_quality(frame)

        # 將影像幀加入快取
        self.frame_cache.add_frame(
            self.gpu_manager.postprocess_from_gpu(frame) if self.use_gpu else frame,
            quality.get_overall_score()
        )

        return frame, quality

    def get_cached_frame(self, timestamp: Optional[float] = None) -> Optional[CachedFrame]:
        """從快取中獲取影像幀

        Args:
            timestamp: 目標時間戳，如果為 None 則返回最新的影像幀

        Returns:
            Optional[CachedFrame]: 快取的影像幀，如果找不到則返回 None
        """
        if timestamp is None:
            return self.frame_cache.get_latest_frame()
        return self.frame_cache.get_frame_by_time(timestamp)

    def generate_frames(self) -> Generator[tuple[np.ndarray, ImageQuality], None, None]:
        """生成影像幀串流

        Yields:
            tuple[numpy.ndarray, ImageQuality]: 影像幀數據和品質評估結果
        """
        while True:
            result = self.get_frame()
            if result is None:
                if not self.try_reconnect():
                    break
                continue

            frame, quality = result

            # 進行基本的影像處理
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            # 如果啟用自動增強
            if settings.AUTO_ENHANCE:
                task_id = self.thread_pool.submit_task(
                    self._enhance_frame,
                    frame,
                    priority=settings.PRIORITY_NORMAL
                )
                try:
                    frame = self.thread_pool.get_result(task_id, timeout=settings.TASK_TIMEOUT)
                except TimeoutError:
                    pass

            yield frame, quality

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """增強影像品質（在執行緒池中執行）

        Args:
            frame: 原始影像幀

        Returns:
            np.ndarray: 增強後的影像幀
        """
        # 如果在 GPU 上，先將影像移到 CPU
        if isinstance(frame, torch.Tensor) and frame.device.type == 'cuda':
            frame = self.gpu_manager.postprocess_from_gpu(frame)

        # 套用 CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=settings.CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 套用銳化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

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
        status = {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "reconnect_attempts": self.reconnect_attempts,
            "fps": self.fps,
            "quality_stats": self.get_quality_stats(),
            "cache_stats": self.frame_cache.get_stats(),
            "gpu_enabled": self.use_gpu
        }

        # 如果使用 GPU，添加 GPU 統計資訊
        if self.use_gpu:
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                current_gpu = gpu_stats[self.device.index]
                status.update({
                    "gpu_memory_used": current_gpu.used_memory,
                    "gpu_memory_total": current_gpu.total_memory,
                    "gpu_temperature": current_gpu.temperature,
                    "gpu_utilization": current_gpu.utilization
                })

        return status