import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# from chatchat.settings import Settings

# FIXME 这里的URL配置需要从Setting中加载获取
engine = create_engine(
    url=os.getenv("DATABASE_URL", "mysql+pymysql://root:Radi2020*@172.18.0.207:7891/remote_llm_server?charset=utf8mb4"),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    """_summary_

    Args:
        DeclarativeBase (_type_): _description_
    """