"""
向量长期记忆：仅存经验、对话摘要、事件（可追加）。

事实性记忆（hobby、name、偏好等）由 FactualStore 热插拔存储，不在此。
使用独立 Chroma 存储，与 RAG 知识库（chroma_db）区分。

写入：为每条记忆写入 created_at、content_sha256；可选按内容去重（见 chroma.yml memory_dedupe_on_write）。

过期数据由 schedule/purge_expired_memory.py 按 memory_ttl_days 定时物理删除；检索侧不做 TTL 过滤。

检索注入：LangChain similarity_search 粗排多取 → 按 created_at 新到旧排序 → 按 content_sha256 去重 → 截断 k 条。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document

from model.factory import embedding_model
from utils.config_utils import chroma_conf
from utils.log_utils import logger
from utils.path_utils import get_abs_path


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _created_at_sort_key(meta: dict | None) -> float:
    """越大越新；无 created_at 的旧数据排后。"""
    if not meta:
        return 0.0
    raw = meta.get("created_at")
    if raw is None:
        return 0.0
    try:
        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _finalize_memory_documents(
        docs: list[Document],
        k: int,
        *,
        dedupe_sha: bool = True,
) -> list[Document]:
    """按 created_at 新→旧排序；可选按 content_sha256 去重（保留时间上最新的一条）。"""
    ordered = sorted(
        docs,
        key=lambda d: _created_at_sort_key(d.metadata or {}),
        reverse=True,
    )
    if not dedupe_sha:
        return ordered[:k]
    seen: set[str] = set()
    out: list[Document] = []
    for d in ordered:
        if not (d.page_content or "").strip():
            continue
        meta = d.metadata or {}
        sha = meta.get("content_sha256")
        if sha:
            if sha in seen:
                continue
            seen.add(sha)
        else:
            h = _content_sha256(d.page_content or "")
            if h in seen:
                continue
            seen.add(h)
        out.append(d)
        if len(out) >= k:
            break
    return out


def _memory_persist_dir() -> str:
    return get_abs_path(chroma_conf.get("memory_persist_directory", "../chroma_memory_db"))


def _memory_collection() -> str:
    return chroma_conf.get("memory_collection_name", "user_memory")


class ChromaMemoryStore:
    """用户个性化记忆的向量存储，供模型上下文注入。"""

    def __init__(self):
        persist_dir = _memory_persist_dir()
        import os
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("[Memory] Chroma persist_directory=%s", persist_dir)
        self._store = Chroma(
            collection_name=_memory_collection(),
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )

    def add(
            self,
            user_id: str,
            content: str,
            memory_type: str = "summary",
            **metadata: str,
    ) -> None:
        """
        添加向量记忆（经验、对话摘要、事件），可追加。

        Args:
            user_id: 用户标识
            content: 文本内容
            memory_type: summary（对话摘要）| experience（经验）| event（事件）
            **metadata: 额外元数据（值须为 Chroma 可接受类型，建议字符串）
        """
        if memory_type not in ("summary", "experience", "event"):
            logger.warning("[Memory] 非预期 memory_type=%s，使用 summary", memory_type)
            memory_type = "summary"
        text = (content or "").strip()
        if not text:
            logger.debug("[Memory] 跳过空内容写入 user_id=%s", user_id)
            return

        created_at = datetime.now(timezone.utc).isoformat()
        sha = _content_sha256(text)
        dedupe = bool(chroma_conf.get("memory_dedupe_on_write", True))
        if dedupe:
            try:
                dup = self._store.get(
                    where={
                        "$and": [
                            {"user_id": {"$eq": user_id}},
                            {"content_sha256": {"$eq": sha}},
                        ]
                    },
                    limit=1,
                )
                if dup.get("ids"):
                    logger.debug("[Memory] 去重跳过写入 user_id=%s type=%s sha=%s...", user_id, memory_type, sha[:12], )
                    return
            except Exception as e:
                logger.warning("[Memory] 去重查询失败，将直接写入: %s", e)

        meta: dict = {
            "user_id": user_id,
            "memory_type": memory_type,
            "created_at": created_at,
            "content_sha256": sha,
        }
        for k, v in metadata.items():
            if k in ("user_id", "memory_type", "created_at", "content_sha256"):
                continue
            meta[k] = v
        doc = Document(page_content=text, metadata=meta)
        self._store.add_documents([doc])

    def get_relevant(
            self,
            user_id: str,
            query: str,
            k: int = 5,
    ) -> list[str]:
        """
        按用户和查询检索相关记忆，返回内容列表。

        流程：similarity_search 粗排多取 → 按 created_at 排序去重 → top k。
        """
        q = (query or "").strip()
        if not q:
            return []

        over = int(chroma_conf.get("memory_retrieve_overfetch", 2))
        cap = int(chroma_conf.get("memory_retrieve_max_cap", 40))
        fetch_n = min(max(k * max(over, 1), k), cap)

        try:
            raw = self._store.similarity_search(
                q,
                k=fetch_n,
                filter={"user_id": {"$eq": user_id}},
            )
        except Exception as e:
            logger.warning("[Memory] similarity_search 失败: %s", e)
            return []

        final_docs = _finalize_memory_documents(raw, k, dedupe_sha=True)
        return [d.page_content for d in final_docs]


_memory_store: ChromaMemoryStore | None = None


def get_memory_store() -> ChromaMemoryStore:
    """获取记忆存储实例（进程内单例，复用 Chroma 连接）。"""
    global _memory_store
    if _memory_store is None:
        _memory_store = ChromaMemoryStore()
    return _memory_store
