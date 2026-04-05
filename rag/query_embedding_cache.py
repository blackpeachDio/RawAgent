"""
检索用查询向量：DashScope 使用 text_type=query 批量请求；LRU 缓存重复问句，降低网络往返。

说明：冷启动首次仍受公网 RTT 与 DashScope 耗时限制；要稳定 <100ms 需本地向量模型或同域部署。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from utils.config_utils import chroma_conf
from utils.log_utils import logger

_CACHE: OrderedDict[str, tuple[float, ...]] = OrderedDict()
_cache_max: int = 0


def _refresh_cache_limit() -> None:
    global _cache_max
    _cache_max = max(0, int(chroma_conf.get("query_embedding_cache_max_entries", 2048) or 0))


def embed_queries_for_retrieval(emb: Any, queries: list[str]) -> list[list[float]]:
    """
    按顺序返回每条 query 的向量；命中缓存的不打远程。
    """
    if not queries:
        return []
    _refresh_cache_limit()
    norm = [(q or "").strip() for q in queries]

    out: list[list[float] | None] = [None] * len(norm)
    miss_i: list[int] = []
    miss_q: list[str] = []

    for i, q in enumerate(norm):
        if _cache_max > 0 and q and q in _CACHE:
            _CACHE.move_to_end(q)
            out[i] = list(_CACHE[q])
        else:
            miss_i.append(i)
            miss_q.append(q if q else " ")

    if miss_q:
        try:
            from langchain_community.embeddings.dashscope import (
                DashScopeEmbeddings,
                embed_with_retry,
            )

            if isinstance(emb, DashScopeEmbeddings):
                batch = embed_with_retry(
                    emb,
                    input=miss_q,
                    text_type="query",
                    model=emb.model,
                )
                vecs = [item["embedding"] for item in batch]
            elif hasattr(emb, "embed_documents"):
                vecs = emb.embed_documents(miss_q)
            else:
                vecs = [emb.embed_query(t) for t in miss_q]
        except Exception as e:
            logger.warning("[RAG] 查询向量批量失败，逐条 embed_query: %s", e)
            vecs = [emb.embed_query(t) for t in miss_q]

        for idx, v in zip(miss_i, vecs):
            out[idx] = v
            qk = norm[idx]
            if _cache_max > 0 and qk:
                _CACHE[qk] = tuple(v)
                _CACHE.move_to_end(qk)
        while _cache_max > 0 and len(_CACHE) > _cache_max:
            _CACHE.popitem(last=False)

    return [x if x is not None else [] for x in out]


def clear_query_embedding_cache() -> None:
    """测试或重建索引后可调用。"""
    _CACHE.clear()
