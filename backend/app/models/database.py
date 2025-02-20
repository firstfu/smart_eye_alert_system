from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..core.config import settings

# 建立資料庫引擎
engine = create_engine(settings.SQLITE_URL)

# 建立 Session 類別
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 建立 Base 類別
Base = declarative_base()

# 取得資料庫連接
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()