from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import time
from dataclasses import dataclass
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """偵測到的物件"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    timestamp: float

class ObjectDetector:
    """物件偵測器"""
    def __init__(self):
        """初始化物件偵測器"""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = []
        self.last_detection_time = 0
        self.detection_interval = settings.DETECTION_INTERVAL
        self.confidence_threshold = settings.DETECTION_CONFIDENCE
        self.model_path = Path(settings.YOLO_MODEL_PATH)

        self._load_model()

    def _load_model(self):
        """載入 YOLO 模型"""
        try:
            if not self.model_path.exists():
                logger.info("下載 YOLOv8n 模型...")
                self.model = YOLO('yolov8n.pt')
            else:
                logger.info(f"載入模型: {self.model_path}")
                self.model = YOLO(self.model_path)

            # 移動模型到指定設備
            self.model.to(self.device)

            logger.info(f"模型載入完成，使用設備: {self.device}")

        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """執行物件偵測

        Args:
            frame: 影像幀

        Returns:
            List[DetectedObject]: 偵測到的物件列表
        """
        current_time = time.time()

        # 檢查是否需要執行偵測
        if current_time - self.last_detection_time < self.detection_interval:
            return []

        try:
            # 執行偵測
            results = self.model(frame, verbose=False)[0]
            detected_objects = []

            # 處理偵測結果
            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detected_object = DetectedObject(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    timestamp=current_time
                )
                detected_objects.append(detected_object)

            self.last_detection_time = current_time
            return detected_objects

        except Exception as e:
            logger.error(f"物件偵測失敗: {str(e)}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """在影像上繪製偵測結果

        Args:
            frame: 原始影像幀
            detections: 偵測到的物件列表

        Returns:
            np.ndarray: 繪製完成的影像幀
        """
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # 繪製邊界框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 準備標籤文字
            label = f"{det.class_name} {det.confidence:.2f}"

            # 計算文字大小
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 繪製標籤背景
            cv2.rectangle(
                result,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # 繪製標籤文字
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return result

    def get_scene_description(self, detections: List[DetectedObject]) -> str:
        """根據偵測結果生成場景描述

        Args:
            detections: 偵測到的物件列表

        Returns:
            str: 場景描述
        """
        if not detections:
            return "場景中未偵測到任何物件"

        # 統計各類物件數量
        object_counts = {}
        for det in detections:
            object_counts[det.class_name] = object_counts.get(det.class_name, 0) + 1

        # 生成描述
        descriptions = []
        for obj_name, count in object_counts.items():
            if count == 1:
                descriptions.append(f"1個{obj_name}")
            else:
                descriptions.append(f"{count}個{obj_name}")

        return f"場景中偵測到：{', '.join(descriptions)}"

    def analyze_spatial_relationships(self, detections: List[DetectedObject]) -> List[str]:
        """分析物件間的空間關係

        Args:
            detections: 偵測到的物件列表

        Returns:
            List[str]: 空間關係描述列表
        """
        if len(detections) < 2:
            return []

        relationships = []
        for i, obj1 in enumerate(detections[:-1]):
            for obj2 in detections[i + 1:]:
                # 計算兩個物件的中心點
                x1_center = (obj1.bbox[0] + obj1.bbox[2]) / 2
                y1_center = (obj1.bbox[1] + obj1.bbox[3]) / 2
                x2_center = (obj2.bbox[0] + obj2.bbox[2]) / 2
                y2_center = (obj2.bbox[1] + obj2.bbox[3]) / 2

                # 判斷相對位置
                if abs(x1_center - x2_center) < abs(y1_center - y2_center):
                    if y1_center < y2_center:
                        relationships.append(f"{obj1.class_name}在{obj2.class_name}的上方")
                    else:
                        relationships.append(f"{obj1.class_name}在{obj2.class_name}的下方")
                else:
                    if x1_center < x2_center:
                        relationships.append(f"{obj1.class_name}在{obj2.class_name}的左側")
                    else:
                        relationships.append(f"{obj1.class_name}在{obj2.class_name}的右側")

        return relationships

    def get_detection_stats(self) -> Dict:
        """獲取偵測統計資訊

        Returns:
            Dict: 統計資訊
        """
        return {
            "model_device": self.device,
            "last_detection_time": self.last_detection_time,
            "detection_interval": self.detection_interval,
            "confidence_threshold": self.confidence_threshold
        }