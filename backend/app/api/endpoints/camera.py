from fastapi import APIRouter, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
from ...services.camera import Camera
from ...services.fall_detector import FallDetector
from ...services.notifier import Notifier
from ...models.database import SessionLocal
from ...models.event import Event

router = APIRouter()
fall_detector = FallDetector()
notifier = Notifier()

@router.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    """串流影像端點"""
    async def generate():
        with Camera(camera_id) as camera:
            for frame in camera.generate_frames():
                # 進行跌倒偵測
                is_fall, confidence, details = fall_detector.detect(frame)

                if is_fall and confidence > 0.7:
                    # 建立事件記錄
                    event_data = {
                        "event_type": "fall_detection",
                        "risk_level": "high",
                        "confidence": confidence,
                        "details": details,
                        "camera_id": camera_id,
                        "location": "未指定"
                    }

                    # 儲存事件
                    db = SessionLocal()
                    try:
                        event = Event(**event_data)
                        db.add(event)
                        db.commit()
                    finally:
                        db.close()

                    # 發送通知
                    await notifier.send_notification(event_data)

                # 轉換影像格式
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """WebSocket 端點，用於即時數據傳輸"""
    await websocket.accept()

    try:
        with Camera(camera_id) as camera:
            for frame in camera.generate_frames():
                # 進行跌倒偵測
                is_fall, confidence, details = fall_detector.detect(frame)

                # 準備回傳資料
                response_data = {
                    "is_fall": is_fall,
                    "confidence": confidence,
                    "details": details
                }

                # 發送結果
                await websocket.send_text(json.dumps(response_data))

    except Exception as e:
        await websocket.close(code=1000)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await websocket.close()