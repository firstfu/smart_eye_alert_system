from fastapi import APIRouter
from .endpoints import camera, cameras

api_router = APIRouter()

# 註冊攝影機相關路由
api_router.include_router(
    camera.router,
    prefix="/camera",
    tags=["camera"]
)

api_router.include_router(
    cameras.router,
    prefix="/cameras",
    tags=["cameras"]
)