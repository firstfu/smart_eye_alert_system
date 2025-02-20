from collections import OrderedDict
import numpy as np
import cv2
from typing import Optional, Dict, Tuple
import time
from dataclasses import dataclass
from ..core.config import settings

@dataclass
class CachedFrame:
    """快取的影像幀資料"""
    frame: np.ndarray  # 原始影像幀
    compressed: bytes  # JPEG 壓縮後的影像數據
    timestamp: float   # 時間戳
    quality_score: float  # 品質分數

class FrameCache:
    def __init__(self, max_size: int = settings.FRAME_CACHE_SIZE):
        """初始化影像快取

        Args:
            max_size: 最大快取大小
        """
        self.max_size = max_size
        self._cache: OrderedDict[float, CachedFrame] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }

    def add_frame(self, frame: np.ndarray, quality_score: float) -> None:
        """添加影像幀到快取

        Args:
            frame: 影像幀數據
            quality_score: 影像品質分數
        """
        # 壓縮影像
        _, compressed = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_QUALITY]
        )

        # 建立快取項目
        timestamp = time.time()
        cached_frame = CachedFrame(
            frame=frame.copy(),
            compressed=compressed.tobytes(),
            timestamp=timestamp,
            quality_score=quality_score
        )

        # 如果快取已滿，移除最舊的項目
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        # 添加新項目
        self._cache[timestamp] = cached_frame

    def get_latest_frame(self) -> Optional[CachedFrame]:
        """獲取最新的影像幀

        Returns:
            Optional[CachedFrame]: 最新的快取影像幀，如果快取為空則返回 None
        """
        self._stats["total_requests"] += 1

        if not self._cache:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        return next(reversed(self._cache.values()))

    def get_frame_by_time(self, timestamp: float, tolerance: float = 0.1) -> Optional[CachedFrame]:
        """根據時間戳獲取影像幀

        Args:
            timestamp: 目標時間戳
            tolerance: 時間容差（秒）

        Returns:
            Optional[CachedFrame]: 符合時間戳的快取影像幀，如果找不到則返回 None
        """
        self._stats["total_requests"] += 1

        # 尋找最接近的時間戳
        closest_timestamp = None
        min_diff = float('inf')

        for ts in self._cache.keys():
            diff = abs(ts - timestamp)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_timestamp = ts

        if closest_timestamp is None:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        return self._cache[closest_timestamp]

    def clear(self) -> None:
        """清空快取"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, float]:
        """獲取快取統計資訊

        Returns:
            Dict[str, float]: 快取統計資料
        """
        total = self._stats["total_requests"]
        if total == 0:
            hit_rate = 0.0
        else:
            hit_rate = self._stats["hits"] / total * 100

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total_requests": total
        }

    def get_frames_in_range(self, start_time: float, end_time: float) -> Dict[float, CachedFrame]:
        """獲取指定時間範圍內的影像幀

        Args:
            start_time: 起始時間
            end_time: 結束時間

        Returns:
            Dict[float, CachedFrame]: 時間戳和影像幀的映射
        """
        return {
            ts: frame for ts, frame in self._cache.items()
            if start_time <= ts <= end_time
        }

    def get_best_quality_frame(self, time_window: float = 1.0) -> Optional[CachedFrame]:
        """獲取最近時間窗口內品質最好的影像幀

        Args:
            time_window: 時間窗口大小（秒）

        Returns:
            Optional[CachedFrame]: 品質最好的影像幀，如果沒有則返回 None
        """
        if not self._cache:
            return None

        current_time = time.time()
        recent_frames = [
            frame for ts, frame in self._cache.items()
            if current_time - ts <= time_window
        ]

        if not recent_frames:
            return None

        return max(recent_frames, key=lambda f: f.quality_score)