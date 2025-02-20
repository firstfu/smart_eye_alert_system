from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import List, Dict
from ...services.camera_manager import CameraManager
from ...services.stream_manager import StreamManager
from ...core.config import settings

router = APIRouter()
camera_manager = CameraManager()
stream_manager = StreamManager()

@router.post("/cameras/{camera_id}")
async def add_camera(camera_id: str):
    """新增攝影機"""
    if await camera_manager.add_camera(camera_id):
        return {"message": f"成功新增攝影機 {camera_id}"}
    raise HTTPException(status_code=400, detail=f"無法新增攝影機 {camera_id}")

@router.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """移除攝影機"""
    if await camera_manager.remove_camera(camera_id):
        return {"message": f"成功移除攝影機 {camera_id}"}
    raise HTTPException(status_code=404, detail=f"攝影機 {camera_id} 不存在")

@router.get("/cameras")
async def list_cameras() -> List[str]:
    """列出所有攝影機"""
    return camera_manager.get_all_cameras()

@router.get("/cameras/status")
async def get_cameras_status() -> Dict[str, Dict]:
    """獲取所有攝影機狀態"""
    return camera_manager.get_all_camera_status()

@router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str) -> Dict:
    """獲取特定攝影機狀態"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"攝影機 {camera_id} 不存在")
    return camera.get_status()

@router.post("/cameras/{camera_id}/reconnect")
async def reconnect_camera(camera_id: str):
    """手動重新連接攝影機"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"攝影機 {camera_id} 不存在")

    if camera.try_reconnect():
        return {"message": f"成功重新連接攝影機 {camera_id}"}
    raise HTTPException(status_code=400, detail=f"無法重新連接攝影機 {camera_id}")

@router.websocket("/ws/cameras/{camera_id}/stream")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """WebSocket 串流端點"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        await websocket.close(code=4004, reason=f"攝影機 {camera_id} 不存在")
        return

    # 檢查連接數量限制
    if (camera_id in stream_manager.active_connections and
        len(stream_manager.active_connections[camera_id]) >= settings.MAX_CONNECTIONS_PER_CAMERA):
        await websocket.close(code=4003, reason="已達到最大連接數限制")
        return

    try:
        await stream_manager.connect(camera_id, websocket)

        # 等待連接關閉
        try:
            while True:
                data = await websocket.receive_text()
                # 這裡可以處理從客戶端接收的訊息
                # 目前我們只是簡單地忽略它們
        except WebSocketDisconnect:
            pass
        finally:
            await stream_manager.disconnect(camera_id, websocket)

    except Exception as e:
        await websocket.close(code=4000, reason=str(e))

@router.on_event("shutdown")
async def shutdown_event():
    """應用程式關閉時的清理工作"""
    await stream_manager.cleanup()
    await camera_manager.cleanup()