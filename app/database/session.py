
import traceback
from contextlib import contextmanager
from functools import wraps
from app.logger import logger
from sqlalchemy.orm import Session

from app.database.base import SessionLocal


@contextmanager
def session_scope() -> Session: 
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        logger.error(f"数据库回滚: {traceback.format_exc()}")
        raise
    finally:
        session.close()


def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper


# def get_db() -> SessionLocal:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# def get_db0() -> SessionLocal:
#     db = SessionLocal()
#     return db
