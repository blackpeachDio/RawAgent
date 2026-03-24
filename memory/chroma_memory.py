"""
模型用长期记忆：历史摘要、用户画像等事实性个性化数据。

使用独立 Chroma 存储，与 RAG 知识库（chroma_db）区分。
支持：事实更新（同 fact_key 覆盖旧值）。
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from model.factory import embedding_model
from utils.config_utils import chroma_conf
from utils.log_utils import logger
from utils.path_utils import get_abs_path


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
            fact_key: str | None = None,
            **metadata: str,
    ) -> None:
        """
        添加记忆片段（摘要、画像、事实等）。

        事实更新：当 fact_key 存在时（如 hobby、city），会先删除该用户下同 key 的旧事实再插入，避免「喜欢苹果」「喜欢香蕉」并存。

        Args:
            user_id: 用户标识
            content: 文本内容
            memory_type: summary | profile | fact
            fact_key: 事实维度标识（如 hobby、city、preference），仅用于 fact 类型；有值时先删除同 key 旧事实
            **metadata: 额外元数据
        """
        # 事实更新：同 fact_key 覆盖旧值（仅 fact 类型）
        if fact_key and memory_type == "fact":
            self._delete_by_fact_key(user_id, fact_key)

        meta = {"user_id": user_id, "memory_type": memory_type, **metadata}
        if fact_key:
            meta["fact_key"] = fact_key
        doc = Document(page_content=content, metadata=meta)
        self._store.add_documents([doc])

    def _delete_by_fact_key(self, user_id: str, fact_key: str) -> None:
        """删除该用户下同 fact_key 的旧事实。"""
        where = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"memory_type": {"$eq": "fact"}},
                {"fact_key": {"$eq": fact_key}},
            ]
        }
        try:
            self._store.delete(where=where)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("[Memory] 删除旧事实失败 user_id=%s fact_key=%s: %s", user_id, fact_key, e)

    def get_relevant(
            self,
            user_id: str,
            query: str,
            k: int = 5,
    ) -> list[str]:
        """
        按用户和查询检索相关记忆，返回内容列表。

        用于注入到模型上下文（如 system prompt 或前置消息）。
        """
        results = self._store.similarity_search(
            query,
            k=k,
            filter={"user_id": {"$eq": user_id}},
        )
        return [d.page_content for d in results]

    def get_relevant_all(
            self,
            user_id: str,
            query: str | None = None,
            k: int = 10,
    ) -> list[str]:
        """
        获取用户最近/相关记忆。query 为空时按 user_id 过滤取最近 k 条。
        """
        if query:
            return self.get_relevant(user_id, query, k=k)
        # 无 query 时按 user_id 取最近添加的（Chroma 默认不保证顺序，这里简化）
        results = self._store.similarity_search(
            f"用户{user_id}的历史摘要与画像",
            k=k,
            filter={"user_id": {"$eq": user_id}},
        )
        return [d.page_content for d in results]


def get_memory_store() -> ChromaMemoryStore:
    """获取记忆存储实例。"""
    return ChromaMemoryStore()
