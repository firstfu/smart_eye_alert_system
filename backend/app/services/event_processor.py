from typing import Dict, List, Optional
import numpy as np
import cv2
import time
from dataclasses import dataclass
from enum import Enum
import logging
from .camera import Camera, ImageQuality
from ..core.config import settings

logger = logging.getLogger(__name__)

class EventType(Enum):
    """事件類型"""
    MOTION_DETECTED = "motion_detected"  # 移動偵測
    QUALITY_ALERT = "quality_alert"      # 品質警告
    CAMERA_OFFLINE = "camera_offline"    # 攝影機離線
    OBJECT_DETECTED = "object_detected"  # 物件偵測
    SCENE_CHANGE = "scene_change"        # 場景變化
    BEHAVIOR_ALERT = "behavior_alert"    # 行為警報

class EventLevel(Enum):
    """事件等級"""
    INFO = "info"           # 一般資訊
    WARNING = "warning"     # 警告
    ERROR = "error"        # 錯誤
    CRITICAL = "critical"  # 緊急

@dataclass
class Event:
    """事件資料"""
    type: EventType
    level: EventLevel
    camera_id: str
    timestamp: float
    details: Dict
    frame_timestamp: Optional[float] = None  # 相關影像幀的時間戳

class ImageProcessor:
    """影像處理器"""
    def __init__(self):
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(
            history=settings.MOTION_HISTORY,
            varThreshold=settings.MOTION_THRESHOLD
        )
        self.last_frame: Optional[np.ndarray] = None
        self.scene_change_threshold = settings.SCENE_CHANGE_THRESHOLD

    def detect_motion(self, frame: np.ndarray) -> tuple[bool, float, np.ndarray]:
        """偵測移動

        Args:
            frame: 影像幀

        Returns:
            tuple[bool, float, np.ndarray]: (是否偵測到移動, 移動程度, 移動遮罩)
        """
        # 轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 套用移動偵測
        mask = self.motion_detector.apply(gray)

        # 計算移動程度
        motion_ratio = np.count_nonzero(mask) / mask.size

        # 判斷是否有移動
        has_motion = motion_ratio > settings.MOTION_RATIO_THRESHOLD

        return has_motion, motion_ratio, mask

    def detect_scene_change(self, frame: np.ndarray) -> tuple[bool, float]:
        """偵測場景變化

        Args:
            frame: 影像幀

        Returns:
            tuple[bool, float]: (是否發生場景變化, 變化程度)
        """
        if self.last_frame is None:
            self.last_frame = frame.copy()
            return False, 0.0

        # 計算目前幀和上一幀的差異
        diff = cv2.absdiff(frame, self.last_frame)
        diff_ratio = np.mean(diff) / 255.0

        # 更新上一幀
        self.last_frame = frame.copy()

        return diff_ratio > self.scene_change_threshold, diff_ratio

    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """增強影像品質

        Args:
            frame: 原始影像幀

        Returns:
            np.ndarray: 增強後的影像幀
        """
        # 自動白平衡
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 銳化
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

class EventProcessor:
    """事件處理器"""
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.events: List[Event] = []
        self.max_events = settings.MAX_EVENTS_HISTORY

    def process_frame(self, camera: Camera, frame: np.ndarray, quality: ImageQuality) -> List[Event]:
        """處理影像幀並產生事件

        Args:
            camera: 攝影機實例
            frame: 影像幀
            quality: 影像品質資料

        Returns:
            List[Event]: 產生的事件列表
        """
        current_events = []
        timestamp = time.time()

        # 檢查影像品質
        if quality.get_overall_score() < settings.MIN_QUALITY_SCORE:
            event = Event(
                type=EventType.QUALITY_ALERT,
                level=EventLevel.WARNING,
                camera_id=camera.camera_id,
                timestamp=timestamp,
                details={
                    "quality_score": quality.get_overall_score(),
                    "brightness": quality.brightness,
                    "contrast": quality.contrast,
                    "blur_score": quality.blur_score
                }
            )
            current_events.append(event)

        # 偵測移動
        has_motion, motion_ratio, motion_mask = self.image_processor.detect_motion(frame)
        if has_motion:
            event = Event(
                type=EventType.MOTION_DETECTED,
                level=EventLevel.INFO,
                camera_id=camera.camera_id,
                timestamp=timestamp,
                details={
                    "motion_ratio": motion_ratio,
                    "location": self._get_motion_location(motion_mask)
                }
            )
            current_events.append(event)

        # 偵測場景變化
        scene_changed, change_ratio = self.image_processor.detect_scene_change(frame)
        if scene_changed:
            event = Event(
                type=EventType.SCENE_CHANGE,
                level=EventLevel.WARNING,
                camera_id=camera.camera_id,
                timestamp=timestamp,
                details={
                    "change_ratio": change_ratio
                }
            )
            current_events.append(event)

        # 儲存事件
        self.events.extend(current_events)

        # 限制事件歷史記錄數量
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        return current_events

    def _get_motion_location(self, motion_mask: np.ndarray) -> Dict[str, int]:
        """計算移動區域位置

        Args:
            motion_mask: 移動遮罩

        Returns:
            Dict[str, int]: 移動區域的邊界框
        """
        # 尋找移動區域的輪廓
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {}

        # 找出最大的移動區域
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }

    def get_events(self,
                  camera_id: Optional[str] = None,
                  event_type: Optional[EventType] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[Event]:
        """獲取事件歷史記錄

        Args:
            camera_id: 攝影機ID過濾
            event_type: 事件類型過濾
            start_time: 起始時間
            end_time: 結束時間

        Returns:
            List[Event]: 符合條件的事件列表
        """
        filtered_events = self.events

        if camera_id:
            filtered_events = [e for e in filtered_events if e.camera_id == camera_id]

        if event_type:
            filtered_events = [e for e in filtered_events if e.type == event_type]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        return filtered_events