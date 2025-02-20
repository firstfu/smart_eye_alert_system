from sqlalchemy import Column, Integer, String, DateTime, Float, JSON
from datetime import datetime
from .database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)  # 例如：'fall_detection', 'crowd_gathering'
    risk_level = Column(String)  # 'high', 'medium', 'low'
    confidence = Column(Float)
    details = Column(JSON)  # 儲存事件的詳細資訊
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String, index=True)
    location = Column(String)
    processed = Column(Integer, default=0)  # 0: 未處理, 1: 已處理
    notification_sent = Column(Integer, default=0)  # 0: 未發送, 1: 已發送