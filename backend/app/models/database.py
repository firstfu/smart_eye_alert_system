from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..core.config import settings

# 創建資料庫引擎
engine = create_engine(
    settings.SQLITE_URL, connect_args={"check_same_thread": False}
)

# 創建 SessionLocal 類別
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 創建 Base 類別
Base = declarative_base()

# 獲取資料庫會話的依賴函數
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()