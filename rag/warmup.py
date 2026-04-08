"""RAG 侧启动预热（与 chroma.yml 配置对齐），供 Agent 入口调用，避免 agent 直接读 chroma_conf。"""
from __future__ import annotations

from utils.config_utils import chroma_conf
from utils.log_utils import logger


def maybe_preload_rerank_cross_encoder() -> None:
    """若开启 rerank 且配置预加载，则加载 CrossEncoder（幂等，内部有缓存）。"""
    if not bool(chroma_conf.get("rerank_preload_on_startup", True)):
        return
    if not bool(chroma_conf.get("rerank_enabled", False)):
        return
    try:
        from rag.retrieval_pipeline import preload_rerank_cross_encoder

        preload_rerank_cross_encoder()
    except Exception as e:
        logger.warning("[RAG] CrossEncoder 预加载未执行: %s", e)
