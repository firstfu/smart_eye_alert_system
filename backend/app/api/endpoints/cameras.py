from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
from ...services.camera_manager import CameraManager
from ...core.config import settings

router = APIRouter()
camera_manager = CameraManager()

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

@router.get("/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: str):
    """獲取攝影機串流（WebSocket）"""
    # 這個端點將在之後實作 WebSocket 串流
    raise HTTPException(status_code=501, detail="此功能尚未實作")