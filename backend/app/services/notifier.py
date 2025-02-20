import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
from ..core.config import settings

class Notifier:
    def __init__(self):
        """初始化通知服務"""
        self.line_notify_url = "https://notify-api.line.me/api/notify"

    async def send_line_notification(self, message: str, image_url: Optional[str] = None) -> bool:
        """發送 LINE Notify 通知

        Args:
            message: 通知訊息
            image_url: 可選的圖片 URL

        Returns:
            bool: 是否發送成功
        """
        if not settings.LINE_NOTIFY_TOKEN:
            return False

        headers = {
            "Authorization": f"Bearer {settings.LINE_NOTIFY_TOKEN}"
        }

        payload = {"message": message}
        if image_url:
            payload["imageFullsize"] = image_url
            payload["imageThumbnail"] = image_url

        try:
            response = requests.post(
                self.line_notify_url,
                headers=headers,
                data=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"LINE Notify 發送失敗: {str(e)}")
            return False

    def format_event_message(self, event_data: Dict[str, Any]) -> str:
        """格式化事件通知訊息

        Args:
            event_data: 事件資料

        Returns:
            str: 格式化後的訊息
        """
        event_time = datetime.fromtimestamp(event_data.get("timestamp", datetime.now().timestamp()))
        formatted_time = event_time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"""
⚠️ 警示通知 ⚠️
時間: {formatted_time}
類型: {event_data.get('event_type', '未知')}
風險等級: {event_data.get('risk_level', '未知')}
位置: {event_data.get('location', '未知')}
置信度: {event_data.get('confidence', 0):.2f}

詳細資訊:
{event_data.get('details', {}).get('message', '無')}
"""
        return message.strip()

    async def send_notification(self, event_data: Dict[str, Any]) -> bool:
        """發送通知

        Args:
            event_data: 事件資料

        Returns:
            bool: 是否發送成功
        """
        message = self.format_event_message(event_data)

        # 發送 LINE 通知
        success = await self.send_line_notification(message)

        return success