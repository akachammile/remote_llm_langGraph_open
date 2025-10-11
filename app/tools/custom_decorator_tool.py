from app.logger import logger as logging
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def retry_decorator(func=None, * , tries=3):
    """设置retry装饰器"""
    def decorator(f):
        @retry(stop=stop_after_attempt(tries),
               wait=wait_exponential(multiplier=1, min=1, max=10),
               retry=retry_if_exception_type(Exception),
               reraise=True)
        def wrapper(*args, **kwargs):
            logging.info(f"调用{f.__name__}方法中")
            return f(*args, **kwargs)
        return wrapper
    
    if func:
        return decorator(func)
    return decorator    