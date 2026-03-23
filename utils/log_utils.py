import contextvars
import logging
import os
from datetime import datetime

from utils.path_utils import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 日志目录
os.makedirs(LOG_ROOT, exist_ok=True)

# 当前请求的 session 标识（用于多会话日志区分）
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_session_id", default=""
)


def set_session_id(session_id: str) -> None:
    """设置当前上下文（线程）的 session_id，后续日志会自动带上此标识。"""
    _session_id_var.set(session_id)


def clear_session_id() -> None:
    """清除当前上下文的 session_id。"""
    try:
        _session_id_var.set("")
    except LookupError:
        pass


def get_session_id() -> str:
    """获取当前上下文的 session_id。"""
    return _session_id_var.get("")


class SessionIdFilter(logging.Filter):
    """为 LogRecord 注入 session_id，供 Formatter 使用。有值时显示 [session_id] 前缀。"""

    def filter(self, record: logging.LogRecord) -> bool:
        sid = get_session_id()
        record.session_id = f"[{sid}] " if sid else ""
        return True


# 日志格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(session_id)s%(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 注入 session_id 的 Filter（用于多会话日志区分）
    logger.addFilter(SessionIdFilter())

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:  # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
