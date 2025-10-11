import sys
from datetime import datetime
from app.cores.config import PROJECT_ROOT
from loguru import logger as _logger


_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """定义日志等级"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (f"{name}_{formatted_date}" if name else formatted_date) # 如果没有提供name，则只使用时间戳
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / f"logs/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level()


# if __name__ == "__main__":
#     logger.info("程序启动")
#     logger.debug("Debug 信息")
#     logger.warning("Warning 信息")
#     logger.error("Error 信息")
#     # logger.critical("Critical message")

#     try:
#         raise ValueError("Test error")
#     except Exception as e:
#         logger.exception(f"An error occurred: {e}")
