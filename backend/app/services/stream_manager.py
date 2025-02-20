import asyncio
import cv2
import base64
import numpy as np
from typing import Dict, Set, Optional, List
from fastapi import WebSocket
import logging
import json
from .camera import Camera
from .event_processor import EventProcessor, Event
from .object_detector import ObjectDetector
from ..core.config import settings
from .behavior_analyzer import BehaviorAnalyzer
import time

logger = logging.getLogger(__name__)

class StreamManager:
    def __init__(self):
        """初始化串流管理器"""
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self.event_processor = EventProcessor()
        self.object_detector = ObjectDetector()
        self.behavior_analyzer = BehaviorAnalyzer()

    async def connect(self, camera_id: str, websocket: WebSocket):
        """建立 WebSocket 連接

        Args:
            camera_id: 攝影機ID
            websocket: WebSocket 連接
        """
        await websocket.accept()

        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()

        self.active_connections[camera_id].add(websocket)

        # 如果是第一個連接，開始串流
        if len(self.active_connections[camera_id]) == 1:
            await self.start_stream(camera_id)

    async def disconnect(self, camera_id: str, websocket: WebSocket):
        """關閉 WebSocket 連接

        Args:
            camera_id: 攝影機ID
            websocket: WebSocket 連接
        """
        if camera_id in self.active_connections:
            self.active_connections[camera_id].remove(websocket)

            # 如果沒有連接了，停止串流
            if not self.active_connections[camera_id]:
                await self.stop_stream(camera_id)
                del self.active_connections[camera_id]

    async def broadcast(self, camera_id: str, message: Dict):
        """廣播訊息給所有連接的客戶端

        Args:
            camera_id: 攝影機ID
            message: 要廣播的訊息
        """
        if camera_id not in self.active_connections:
            return

        # 取得需要移除的斷開連接
        disconnected = set()
        message_json = json.dumps(message)

        for websocket in self.active_connections[camera_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"廣播訊息失敗: {str(e)}")
                disconnected.add(websocket)

        # 移除斷開的連接
        for websocket in disconnected:
            await self.disconnect(camera_id, websocket)

    async def broadcast_event(self, camera_id: str, event: Event):
        """廣播事件給所有連接的客戶端

        Args:
            camera_id: 攝影機ID
            event: 事件資料
        """
        event_data = {
            "type": "event",
            "event_type": event.type.value,
            "level": event.level.value,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp,
            "details": event.details
        }
        await self.broadcast(camera_id, event_data)

    async def start_stream(self, camera_id: str):
        """開始影像串流

        Args:
            camera_id: 攝影機ID
        """
        if camera_id in self._stream_tasks:
            return

        async def stream_loop(camera: Camera):
            while True:
                try:
                    # 檢查是否還有連接
                    if not self.active_connections.get(camera_id):
                        break

                    # 獲取影像幀
                    result = camera.get_frame()
                    if result is None:
                        await asyncio.sleep(0.1)
                        continue

                    frame, quality = result

                    # 執行物件偵測
                    detections = self.object_detector.detect(frame)

                    # 執行行為分析
                    if settings.BEHAVIOR_TRACKING_ENABLED and detections:
                        behavior_results = self.behavior_analyzer.update(detections)

                        # 處理行為分析警報
                        for alert in behavior_results["alerts"]:
                            event = Event(
                                type=EventType.BEHAVIOR_ALERT,
                                level=EventLevel.WARNING,
                                camera_id=camera_id,
                                timestamp=time.time(),
                                details={
                                    "alert_type": alert["type"],
                                    "object_id": alert["object_id"],
                                    "location": alert["location"],
                                    **alert
                                }
                            )
                            await self.broadcast_event(camera_id, event)

                    # 生成場景描述
                    scene_description = self.object_detector.get_scene_description(detections)
                    spatial_relationships = self.object_detector.analyze_spatial_relationships(detections)

                    # 在影像上繪製偵測結果
                    if settings.DRAW_DETECTIONS and detections:
                        frame = self.object_detector.draw_detections(frame, detections)

                    # 處理影像幀並產生事件
                    events = self.event_processor.process_frame(camera, frame, quality)

                    # 廣播事件
                    for event in events:
                        await self.broadcast_event(camera_id, event)

                    # 影像增強（如果啟用）
                    if settings.AUTO_ENHANCE:
                        frame = self.event_processor.image_processor.enhance_image(frame)

                    # 從快取中獲取壓縮後的影像
                    cached_frame = camera.get_cached_frame()
                    if cached_frame is None:
                        await asyncio.sleep(0.1)
                        continue

                    # 準備串流資料
                    stream_data = {
                        "type": "frame",
                        "camera_id": camera_id,
                        "image": base64.b64encode(cached_frame.compressed).decode('utf-8'),
                        "timestamp": cached_frame.timestamp,
                        "quality": {
                            "brightness": quality.brightness,
                            "contrast": quality.contrast,
                            "blur_score": quality.blur_score,
                            "overall_score": quality.get_overall_score()
                        },
                        "detections": [
                            {
                                "class_name": det.class_name,
                                "confidence": det.confidence,
                                "bbox": det.bbox
                            }
                            for det in detections
                        ],
                        "scene_description": scene_description,
                        "spatial_relationships": spatial_relationships,
                        "behavior_analysis": {
                            "tracked_objects": behavior_results["tracked_objects"],
                            "interactions": behavior_results["interactions"]
                        }
                    }

                    # 廣播影像幀
                    await self.broadcast(camera_id, stream_data)

                    # 控制串流速率
                    await asyncio.sleep(1.0 / settings.STREAM_FPS)

                except Exception as e:
                    logger.error(f"串流處理發生錯誤: {str(e)}")
                    await asyncio.sleep(1.0)

        self._stream_tasks[camera_id] = asyncio.create_task(stream_loop(camera_id))

    async def stop_stream(self, camera_id: str):
        """停止影像串流

        Args:
            camera_id: 攝影機ID
        """
        if camera_id in self._stream_tasks:
            self._stream_tasks[camera_id].cancel()
            try:
                await self._stream_tasks[camera_id]
            except asyncio.CancelledError:
                pass
            del self._stream_tasks[camera_id]

    async def cleanup(self):
        """清理所有資源"""
        for camera_id in list(self._stream_tasks.keys()):
            await self.stop_stream(camera_id)

    def get_events(self,
                  camera_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[Event]:
        """獲取事件歷史記錄

        Args:
            camera_id: 攝影機ID
            start_time: 起始時間
            end_time: 結束時間

        Returns:
            List[Event]: 事件列表
        """
        return self.event_processor.get_events(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time
        )