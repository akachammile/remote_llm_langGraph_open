from datetime import datetime, UTC

from sqlalchemy import Column, DateTime, Integer, String


class BaseModel:
    """
    基础模型类，便于扩展
    """
    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    create_time = Column(DateTime, default=datetime.now(UTC), comment="创建时间")
    update_time = Column(DateTime, default=None, onupdate=datetime.now(UTC), comment="更新时间")
    create_by = Column(String, default=None, comment="创建者")
    update_by = Column(String, default=None, comment="更新者")
