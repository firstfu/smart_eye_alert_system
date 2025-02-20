import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
import logging
from dataclasses import dataclass
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class PoseKeypoint:
    """姿態關鍵點"""
    x: float  # 標準化的 x 座標 (0-1)
    y: float  # 標準化的 y 座標 (0-1)
    z: float  # 相對深度
    visibility: float  # 可見度分數
    name: str  # 關鍵點名稱

@dataclass
class PoseResult:
    """姿態估計結果"""
    landmarks: np.ndarray  # 關鍵點座標
    confidence: float  # 置信度
    world_landmarks: np.ndarray  # 3D世界座標系中的關鍵點

class PoseEstimator:
    """姿態估計器類別"""
    def __init__(self):
        """初始化姿態估計器"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=settings.POSE_STATIC_IMAGE_MODE,
            model_complexity=settings.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=settings.POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.POSE_TRACKING_CONFIDENCE,
            smooth_landmarks=settings.POSE_SMOOTH_LANDMARKS
        )

    def detect(self, frame: np.ndarray) -> Optional[PoseResult]:
        """偵測影像中的人體姿態

        Args:
            frame: BGR格式的影像幀

        Returns:
            Optional[PoseResult]: 姿態估計結果，如果沒有偵測到則返回None
        """
        # 轉換為RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 執行姿態估計
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # 轉換關鍵點為numpy陣列
        landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                            for lm in results.pose_landmarks.landmark])
        world_landmarks = np.array([[lm.x, lm.y, lm.z]
                                  for lm in results.pose_world_landmarks.landmark])

        # 計算整體置信度
        confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

        return PoseResult(
            landmarks=landmarks,
            confidence=confidence,
            world_landmarks=world_landmarks
        )

    def analyze_pose(self, pose_result: PoseResult) -> Dict[str, float]:
        """分析姿態特徵

        Args:
            pose_result: 姿態估計結果

        Returns:
            Dict[str, float]: 姿態特徵字典
        """
        features = {}

        # 計算身體傾斜角度
        features["body_tilt"] = self._calculate_body_tilt(pose_result.world_landmarks)

        # 計算身體高度比例
        features["height_ratio"] = self._calculate_height_ratio(pose_result.landmarks)

        # 計算穩定性分數
        features["stability_score"] = self._calculate_stability(pose_result.world_landmarks)

        # 計算運動速度
        features["movement_speed"] = self._calculate_movement_speed(pose_result.landmarks)

        return features

    def _calculate_body_tilt(self, world_landmarks: np.ndarray) -> float:
        """計算身體傾斜角度

        Args:
            world_landmarks: 3D世界座標系中的關鍵點

        Returns:
            float: 身體傾斜角度（度）
        """
        # 使用肩部中點和臀部中點計算身體軸向
        left_shoulder = world_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = world_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = world_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = world_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2

        # 計算身體向量
        body_vector = shoulder_mid - hip_mid

        # 計算與垂直軸的角度
        vertical_vector = np.array([0, 1, 0])
        angle = np.arccos(np.dot(body_vector, vertical_vector) /
                         (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector)))

        return np.degrees(angle)

    def _calculate_height_ratio(self, landmarks: np.ndarray) -> float:
        """計算身體高度比例

        Args:
            landmarks: 2D影像座標系中的關鍵點

        Returns:
            float: 高度比例（當前高度/標準站立高度）
        """
        # 使用肩部和臀部關鍵點計算當前高度
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2

        current_height = abs(shoulder_mid_y - hip_mid_y)

        # 使用預設的標準高度（可以根據歷史數據動態調整）
        standard_height = 0.5  # 假設標準高度為影像高度的50%

        return current_height / standard_height

    def _calculate_stability(self, world_landmarks: np.ndarray) -> float:
        """計算姿態穩定性分數

        Args:
            world_landmarks: 3D世界座標系中的關鍵點

        Returns:
            float: 穩定性分數（0-1，1表示最穩定）
        """
        # 計算關鍵點的分佈情況
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

        points = world_landmarks[key_points]

        # 計算點雲的協方差矩陣
        covariance = np.cov(points.T)

        # 使用特徵值分析穩定性
        eigenvalues = np.linalg.eigvals(covariance)

        # 計算穩定性分數（特徵值的比例）
        stability = min(eigenvalues) / max(eigenvalues)

        return float(stability)

    def _calculate_movement_speed(self, landmarks: np.ndarray) -> float:
        """計算運動速度

        Args:
            landmarks: 2D影像座標系中的關鍵點

        Returns:
            float: 運動速度分數
        """
        # 計算主要關節點的平均位移
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

        points = landmarks[key_points]

        # 計算位移的標準差作為速度指標
        movement = np.std(points[:, :2])  # 只考慮x,y座標

        return float(movement)

    def draw_pose(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """在影像上繪製姿態估計結果

        Args:
            frame: BGR格式的影像幀
            pose_result: 姿態估計結果

        Returns:
            np.ndarray: 繪製後的影像
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # 建立姿態關鍵點結果物件
        landmarks = mp.solutions.pose.PoseLandmark
        pose_landmarks = mp.solutions.pose.PoseLandmark._member_map_

        # 轉換關鍵點格式
        landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for landmark in pose_result.landmarks:
            landmark_proto = landmarks_proto.landmark.add()
            landmark_proto.x = landmark[0]
            landmark_proto.y = landmark[1]
            landmark_proto.z = landmark[2]
            landmark_proto.visibility = landmark[3]

        # 繪製骨架
        mp_drawing.draw_landmarks(
            frame,
            landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        return frame