"""RAG 侧启动预热（与 chroma.yml 配置对齐），供 Agent 入口调用，避免 agent 直接读 chroma_conf。"""
from __future__ import annotations

from utils.config_utils import chroma_conf
from utils.log_utils import logger


def maybe_preload_rerank_cross_encoder() -> None:
    """
    兼容旧入口名：精排已改为百炼在线 API，无需本地预加载模型。
    rerank_preload_on_startup 保留为无操作，避免改 Agent 初始化代码。
    """
    if not bool(chroma_conf.get("rerank_preload_on_startup", True)):
        return
    if not bool(chroma_conf.get("rerank_enabled", False)):
        return
    logger.debug("[RAG] rerank 使用百炼 DashScope API，跳过本地预加载")
