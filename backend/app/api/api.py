from fastapi import APIRouter
from .endpoints import camera

api_router = APIRouter()

# 註冊攝影機相關路由
api_router.include_router(
    camera.router,
    prefix="/camera",
    tags=["camera"]
)