from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)
    name = Column(String)
    rtsp_url = Column(String)
    username = Column(String)
    password = Column(String)
    location = Column(String)
    status = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())