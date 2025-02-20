from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
from .core.config import settings
from .services.camera_manager import CameraManager
from .services.stream_manager import StreamManager
from .db.session import get_db
from .api.api import api_router
from .models.database import Base, engine
from .models.camera import Camera

# 創建資料庫表格
Base.metadata.create_all(bind=engine)

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Smart Eye Alert System",
    version="1.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設定具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_manager = StreamManager()
camera_manager = CameraManager()

# 註冊 API 路由
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """系統啟動時執行"""
    try:
        # 1. 從資料庫讀取所有攝影機設定
        db = next(get_db())
        try:
            cameras = db.query(Camera).filter(Camera.status == 'active').all()
            logger.info(f"找到 {len(cameras)} 個活動中的攝影機")

            # 2. 逐一啟動攝影機串流
            for camera in cameras:
                try:
                    # 初始化攝影機連接
                    await camera_manager.add_camera(
                        camera_id=camera.id,
                        rtsp_url=camera.rtsp_url,
                        username=camera.username,
                        password=camera.password
                    )

                    # 啟動串流處理
                    await stream_manager.start_stream(camera.id)

                    logger.info(f"成功啟動攝影機 {camera.id} 的監測")
                except Exception as e:
                    logger.error(f"啟動攝影機 {camera.id} 失敗: {str(e)}")
                    continue

        finally:
            db.close()

    except Exception as e:
        logger.error(f"系統啟動失敗: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """系統關閉時執行"""
    try:
        # 清理資源
        await stream_manager.cleanup()
        await camera_manager.cleanup()
        logger.info("系統成功關閉")
    except Exception as e:
        logger.error(f"系統關閉時發生錯誤: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Smart Eye Alert System API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }