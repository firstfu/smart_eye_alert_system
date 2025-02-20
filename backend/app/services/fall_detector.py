import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Any
import math
from ..core.config import settings

class FallDetector:
    def __init__(self):
        """初始化跌倒偵測器"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=settings.DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.TRACKING_CONFIDENCE
        )

        # 用於追蹤前一幀的姿態
        self.prev_pose_landmarks = None
        self.fall_threshold = 45  # 度數閾值

    def detect(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """偵測影像中是否有跌倒情況

        Args:
            frame: 輸入影像幀

        Returns:
            Tuple[bool, float, Dict]: (是否跌倒, 置信度, 詳細資訊)
        """
        # 轉換顏色空間
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if not results.pose_landmarks:
            return False, 0.0, {"message": "未檢測到人體"}

        # 計算身體傾斜角度
        angle = self._calculate_body_angle(results.pose_landmarks)

        # 計算移動速度（如果有前一幀資料）
        movement_speed = self._calculate_movement_speed(results.pose_landmarks)

        # 更新前一幀的姿態資料
        self.prev_pose_landmarks = results.pose_landmarks

        # 判斷是否跌倒
        is_fall = angle > self.fall_threshold and movement_speed > 0.5
        confidence = self._calculate_confidence(angle, movement_speed)

        details = {
            "angle": angle,
            "movement_speed": movement_speed,
            "landmarks": self._get_key_points(results.pose_landmarks)
        }

        return is_fall, confidence, details

    def _calculate_body_angle(self, landmarks) -> float:
        """計算身體傾斜角度"""
        # 使用髖關節和肩關節的中點來計算身體軸線
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 計算中點
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2

        # 計算角度
        angle = math.degrees(math.atan2(
            abs(shoulder_mid_x - hip_mid_x),
            abs(shoulder_mid_y - hip_mid_y)
        ))

        return angle

    def _calculate_movement_speed(self, current_landmarks) -> float:
        """計算姿態變化速度"""
        if self.prev_pose_landmarks is None:
            return 0.0

        # 計算關鍵點的平均位移
        total_displacement = 0
        num_landmarks = 0

        for i in [self.mp_pose.PoseLandmark.LEFT_HIP,
                 self.mp_pose.PoseLandmark.RIGHT_HIP,
                 self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                 self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
            curr = current_landmarks.landmark[i]
            prev = self.prev_pose_landmarks.landmark[i]

            displacement = math.sqrt(
                (curr.x - prev.x) ** 2 +
                (curr.y - prev.y) ** 2
            )
            total_displacement += displacement
            num_landmarks += 1

        return total_displacement / num_landmarks if num_landmarks > 0 else 0.0

    def _calculate_confidence(self, angle: float, movement_speed: float) -> float:
        """計算偵測結果的置信度"""
        angle_factor = min(1.0, angle / self.fall_threshold)
        speed_factor = min(1.0, movement_speed / 0.5)

        return (angle_factor + speed_factor) / 2

    def _get_key_points(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """獲取關鍵點座標"""
        key_points = {}
        for landmark in self.mp_pose.PoseLandmark:
            point = landmarks.landmark[landmark]
            key_points[landmark.name] = (point.x, point.y)

        return key_points