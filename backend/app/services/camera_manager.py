from typing import Dict, Optional, List
from .camera import Camera, ImageQuality
import asyncio
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        """初始化攝影機管理器"""
        self.cameras: Dict[str, Camera] = {}
        self._quality_check_task = None
        self._connection_check_task = None

    async def add_camera(self, camera_id: str) -> bool:
        """新增攝影機

        Args:
            camera_id: 攝影機ID

        Returns:
            bool: 是否成功新增
        """
        try:
            if camera_id in self.cameras:
                logger.warning(f"攝影機 {camera_id} 已存在")
                return False

            if len(self.cameras) >= settings.MAX_CAMERAS:
                logger.error("已達到最大攝影機數量限制")
                return False

            camera = Camera(camera_id)
            if not camera.start():
                raise RuntimeError("無法啟動攝影機")

            self.cameras[camera_id] = camera

            # 確認攝影機可以正常獲取影像
            test_frame = camera.get_frame()
            if test_frame is None:
                raise RuntimeError("無法從攝影機獲取影像")

            # 如果這是第一個攝影機，啟動監控任務
            if len(self.cameras) == 1:
                await self.start_monitoring()

            logger.info(f"成功新增攝影機 {camera_id}")
            return True

        except Exception as e:
            logger.error(f"新增攝影機 {camera_id} 失敗: {str(e)}")
            if camera_id in self.cameras:
                await self.remove_camera(camera_id)
            return False

    async def remove_camera(self, camera_id: str) -> bool:
        """移除攝影機

        Args:
            camera_id: 攝影機ID

        Returns:
            bool: 是否成功移除
        """
        try:
            if camera_id not in self.cameras:
                logger.warning(f"攝影機 {camera_id} 不存在")
                return False

            camera = self.cameras[camera_id]
            camera.stop()
            del self.cameras[camera_id]

            # 如果沒有攝影機了，停止監控任務
            if not self.cameras:
                await self.stop_monitoring()

            logger.info(f"成功移除攝影機 {camera_id}")
            return True

        except Exception as e:
            logger.error(f"移除攝影機 {camera_id} 失敗: {str(e)}")
            return False

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """獲取攝影機實例

        Args:
            camera_id: 攝影機ID

        Returns:
            Optional[Camera]: 攝影機實例，如果不存在則返回 None
        """
        return self.cameras.get(camera_id)

    def get_all_cameras(self) -> List[str]:
        """獲取所有攝影機ID

        Returns:
            List[str]: 攝影機ID列表
        """
        return list(self.cameras.keys())

    def get_all_camera_status(self) -> Dict[str, Dict]:
        """獲取所有攝影機狀態

        Returns:
            Dict[str, Dict]: 攝影機狀態字典
        """
        return {
            camera_id: camera.get_status()
            for camera_id, camera in self.cameras.items()
        }

    async def start_monitoring(self):
        """啟動所有監控任務"""
        await self.start_quality_monitoring()
        await self.start_connection_monitoring()

    async def stop_monitoring(self):
        """停止所有監控任務"""
        await self.stop_quality_monitoring()
        await self.stop_connection_monitoring()

    async def start_quality_monitoring(self):
        """開始品質監控"""
        if self._quality_check_task is not None:
            return

        async def quality_check_loop():
            while True:
                try:
                    for camera_id, camera in self.cameras.items():
                        stats = camera.get_quality_stats()

                        # 檢查影像品質是否符合要求
                        if stats["avg_brightness"] < settings.MIN_BRIGHTNESS:
                            logger.warning(f"攝影機 {camera_id} 亮度過低: {stats['avg_brightness']:.2f}")

                        if stats["avg_contrast"] < settings.MIN_CONTRAST:
                            logger.warning(f"攝影機 {camera_id} 對比度過低: {stats['avg_contrast']:.2f}")

                        if stats["avg_blur_score"] < settings.MIN_BLUR_SCORE:
                            logger.warning(f"攝影機 {camera_id} 影像模糊: {stats['avg_blur_score']:.2f}")

                        if stats["fps"] < settings.MIN_FPS:
                            logger.warning(f"攝影機 {camera_id} FPS 過低: {stats['fps']}")

                except Exception as e:
                    logger.error(f"品質監控發生錯誤: {str(e)}")

                await asyncio.sleep(settings.QUALITY_CHECK_INTERVAL)

        self._quality_check_task = asyncio.create_task(quality_check_loop())

    async def start_connection_monitoring(self):
        """開始連接狀態監控"""
        if self._connection_check_task is not None:
            return

        async def connection_check_loop():
            while True:
                try:
                    for camera_id, camera in self.cameras.items():
                        if not camera.is_connected:
                            logger.warning(f"攝影機 {camera_id} 連接中斷，嘗試重新連接")
                            if camera.try_reconnect():
                                logger.info(f"攝影機 {camera_id} 重新連接成功")
                            else:
                                logger.error(f"攝影機 {camera_id} 重新連接失敗")

                except Exception as e:
                    logger.error(f"連接監控發生錯誤: {str(e)}")

                await asyncio.sleep(1.0)  # 每秒檢查一次連接狀態

        self._connection_check_task = asyncio.create_task(connection_check_loop())

    async def stop_quality_monitoring(self):
        """停止品質監控"""
        if self._quality_check_task is not None:
            self._quality_check_task.cancel()
            try:
                await self._quality_check_task
            except asyncio.CancelledError:
                pass
            self._quality_check_task = None

    async def stop_connection_monitoring(self):
        """停止連接監控"""
        if self._connection_check_task is not None:
            self._connection_check_task.cancel()
            try:
                await self._connection_check_task
            except asyncio.CancelledError:
                pass
            self._connection_check_task = None

    async def cleanup(self):
        """清理所有資源"""
        await self.stop_monitoring()
        for camera_id in list(self.cameras.keys()):
            await self.remove_camera(camera_id)