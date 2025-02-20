from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import cv2
from dataclasses import dataclass
import time
from collections import deque
import logging
from .object_detector import DetectedObject
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """追蹤物件資訊"""
    id: int
    class_name: str
    trajectory: deque  # 儲存物件軌跡
    first_seen: float
    last_seen: float
    current_bbox: Tuple[int, int, int, int]
    velocity: Tuple[float, float]  # (dx, dy)
    state: str  # 'moving', 'stationary', 'fallen'

@dataclass
class Interaction:
    """物件互動資訊"""
    object1_id: int
    object2_id: int
    interaction_type: str
    start_time: float
    distance: float
    duration: float

class BehaviorAnalyzer:
    """行為分析器"""
    def __init__(self):
        """初始化行為分析器"""
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.max_trajectory_length = 50
        self.interactions: List[Interaction] = []
        self.stationary_threshold = 0.1  # 靜止閾值
        self.interaction_distance_threshold = 100  # 互動距離閾值（像素）
        self.fallen_angle_threshold = 45  # 跌倒角度閾值

    def update(self, detections: List[DetectedObject]) -> Dict:
        """更新物件追蹤和行為分析

        Args:
            detections: 偵測到的物件列表

        Returns:
            Dict: 分析結果
        """
        current_time = time.time()
        current_objects: Set[int] = set()
        analysis_results = {
            "tracked_objects": [],
            "interactions": [],
            "alerts": []
        }

        # 更新追蹤物件
        for detection in detections:
            matched_id = self._match_detection(detection)
            if matched_id is None:
                # 新物件
                matched_id = self.next_id
                self.next_id += 1
                self.tracked_objects[matched_id] = TrackedObject(
                    id=matched_id,
                    class_name=detection.class_name,
                    trajectory=deque(maxlen=self.max_trajectory_length),
                    first_seen=current_time,
                    last_seen=current_time,
                    current_bbox=detection.bbox,
                    velocity=(0, 0),
                    state='moving'
                )
            else:
                # 更新現有物件
                tracked_obj = self.tracked_objects[matched_id]
                old_center = self._get_bbox_center(tracked_obj.current_bbox)
                new_center = self._get_bbox_center(detection.bbox)

                # 計算速度
                dt = current_time - tracked_obj.last_seen
                if dt > 0:
                    dx = (new_center[0] - old_center[0]) / dt
                    dy = (new_center[1] - old_center[1]) / dt
                    tracked_obj.velocity = (dx, dy)

                # 更新狀態
                speed = (dx**2 + dy**2)**0.5
                if speed < self.stationary_threshold:
                    tracked_obj.state = 'stationary'
                else:
                    tracked_obj.state = 'moving'

                # 更新位置和時間
                tracked_obj.current_bbox = detection.bbox
                tracked_obj.last_seen = current_time
                tracked_obj.trajectory.append(new_center)

            current_objects.add(matched_id)

        # 分析互動
        self._analyze_interactions(current_objects, current_time)

        # 分析異常行為
        alerts = self._analyze_abnormal_behaviors(current_objects)
        if alerts:
            analysis_results["alerts"].extend(alerts)

        # 整理分析結果
        for obj_id in current_objects:
            obj = self.tracked_objects[obj_id]
            analysis_results["tracked_objects"].append({
                "id": obj.id,
                "class_name": obj.class_name,
                "state": obj.state,
                "position": self._get_bbox_center(obj.current_bbox),
                "velocity": obj.velocity,
                "duration": current_time - obj.first_seen
            })

        for interaction in self.interactions:
            if interaction.start_time + interaction.duration >= current_time:
                analysis_results["interactions"].append({
                    "type": interaction.interaction_type,
                    "objects": [interaction.object1_id, interaction.object2_id],
                    "duration": interaction.duration,
                    "distance": interaction.distance
                })

        return analysis_results

    def _match_detection(self, detection: DetectedObject) -> Optional[int]:
        """匹配偵測結果與已追蹤物件

        Args:
            detection: 偵測到的物件

        Returns:
            Optional[int]: 匹配的物件ID，如果是新物件則返回 None
        """
        min_dist = float('inf')
        matched_id = None
        detection_center = self._get_bbox_center(detection.bbox)

        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.class_name != detection.class_name:
                continue

            tracked_center = self._get_bbox_center(tracked_obj.current_bbox)
            dist = ((detection_center[0] - tracked_center[0])**2 +
                   (detection_center[1] - tracked_center[1])**2)**0.5

            if dist < min_dist:
                min_dist = dist
                matched_id = obj_id

        return matched_id if min_dist < 100 else None

    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """計算邊界框中心點

        Args:
            bbox: 邊界框座標 (x1, y1, x2, y2)

        Returns:
            Tuple[float, float]: 中心點座標 (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _analyze_interactions(self, current_objects: Set[int], current_time: float):
        """分析物件間的互動

        Args:
            current_objects: 當前追蹤的物件ID集合
            current_time: 當前時間
        """
        # 清理過期的互動記錄
        self.interactions = [i for i in self.interactions
                           if i.start_time + i.duration >= current_time - 5.0]

        # 分析物件間的互動
        for obj1_id in current_objects:
            obj1 = self.tracked_objects[obj1_id]
            for obj2_id in current_objects:
                if obj1_id >= obj2_id:
                    continue

                obj2 = self.tracked_objects[obj2_id]
                center1 = self._get_bbox_center(obj1.current_bbox)
                center2 = self._get_bbox_center(obj2.current_bbox)
                distance = ((center1[0] - center2[0])**2 +
                          (center1[1] - center2[1])**2)**0.5

                if distance < self.interaction_distance_threshold:
                    # 檢查是否已存在互動記錄
                    existing_interaction = None
                    for interaction in self.interactions:
                        if {interaction.object1_id, interaction.object2_id} == {obj1_id, obj2_id}:
                            existing_interaction = interaction
                            break

                    if existing_interaction:
                        existing_interaction.duration = current_time - existing_interaction.start_time
                        existing_interaction.distance = distance
                    else:
                        # 建立新的互動記錄
                        interaction_type = self._determine_interaction_type(obj1, obj2)
                        self.interactions.append(Interaction(
                            object1_id=obj1_id,
                            object2_id=obj2_id,
                            interaction_type=interaction_type,
                            start_time=current_time,
                            distance=distance,
                            duration=0.0
                        ))

    def _determine_interaction_type(self, obj1: TrackedObject, obj2: TrackedObject) -> str:
        """判斷互動類型

        Args:
            obj1: 第一個物件
            obj2: 第二個物件

        Returns:
            str: 互動類型描述
        """
        # 根據物件類型和狀態判斷互動類型
        if obj1.state == 'fallen' or obj2.state == 'fallen':
            return 'emergency_assistance'
        elif obj1.state == 'stationary' and obj2.state == 'stationary':
            return 'gathering'
        else:
            return 'passing_by'

    def _analyze_abnormal_behaviors(self, current_objects: Set[int]) -> List[Dict]:
        """分析異常行為

        Args:
            current_objects: 當前追蹤的物件ID集合

        Returns:
            List[Dict]: 異常行為警報列表
        """
        alerts = []
        current_time = time.time()

        for obj_id in current_objects:
            obj = self.tracked_objects[obj_id]

            # 檢查是否跌倒
            if obj.class_name == 'person' and len(obj.trajectory) >= 2:
                # 計算垂直移動速度
                last_pos = obj.trajectory[-1]
                first_pos = obj.trajectory[0]
                vertical_speed = abs(last_pos[1] - first_pos[1]) / (obj.last_seen - obj.first_seen)

                # 檢查姿態變化（透過邊界框的寬高比）
                x1, y1, x2, y2 = obj.current_bbox
                aspect_ratio = (y2 - y1) / (x2 - x1)

                if vertical_speed > 2.0 and aspect_ratio < 0.8:  # 快速下落且姿態異常
                    obj.state = 'fallen'
                    alerts.append({
                        "type": "fall_detected",
                        "object_id": obj.id,
                        "confidence": 0.8,
                        "timestamp": current_time,
                        "location": obj.current_bbox
                    })

            # 檢查異常停留
            if obj.state == 'stationary':
                stationary_duration = current_time - obj.last_seen
                if stationary_duration > 300:  # 停留超過5分鐘
                    alerts.append({
                        "type": "long_stationary",
                        "object_id": obj.id,
                        "duration": stationary_duration,
                        "timestamp": current_time,
                        "location": obj.current_bbox
                    })

            # 檢查異常移動
            if obj.state == 'moving':
                speed = (obj.velocity[0]**2 + obj.velocity[1]**2)**0.5
                if speed > 5.0:  # 異常快速移動
                    alerts.append({
                        "type": "rapid_movement",
                        "object_id": obj.id,
                        "speed": speed,
                        "timestamp": current_time,
                        "location": obj.current_bbox
                    })

        return alerts

    def get_object_history(self, object_id: int) -> Optional[Dict]:
        """獲取物件的歷史記錄

        Args:
            object_id: 物件ID

        Returns:
            Optional[Dict]: 物件的歷史記錄
        """
        if object_id not in self.tracked_objects:
            return None

        obj = self.tracked_objects[object_id]
        return {
            "id": obj.id,
            "class_name": obj.class_name,
            "trajectory": list(obj.trajectory),
            "first_seen": obj.first_seen,
            "last_seen": obj.last_seen,
            "state": obj.state,
            "total_distance": self._calculate_trajectory_distance(obj.trajectory)
        }

    def _calculate_trajectory_distance(self, trajectory: deque) -> float:
        """計算軌跡總距離

        Args:
            trajectory: 軌跡點列表

        Returns:
            float: 總距離
        """
        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            total_distance += distance

        return total_distance