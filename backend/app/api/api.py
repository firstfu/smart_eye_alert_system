from fastapi import APIRouter
from .endpoints import camera, cameras, events, analysis

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

# 註冊事件相關路由
api_router.include_router(
    events.router,
    prefix="/events",
    tags=["events"]
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["analysis"]
)