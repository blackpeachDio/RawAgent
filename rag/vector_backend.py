"""
RAG 向量库后端抽象：屏蔽 Chroma / Milvus 等具体实现，供 ``rag/offline_index``、
``rag/online_query``、``rag/retrieval_pipeline`` 统一调用。

设计原则：
- 对调用方只暴露语义化方法（add_documents / search_by_vectors / similarity_search / dump_all / embedding），
  不再依赖具体库的私有属性（例如旧实现用过的 ``vector_store._collection`` / ``vector_store._embedding_function``）。
- 后端按 ``chroma_conf["vector_backend"]`` 选择，默认 ``chroma``；切换到 ``milvus`` 时需额外安装
  ``langchain-milvus`` 与 ``pymilvus``，并在 ``config/chroma.yml`` 配置 ``milvus_*`` 键。
- Milvus 后端仅用于 RAG 知识库；用户记忆向量库（``memory/chroma_memory.py``）维持 Chroma 不变。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.config_utils import chroma_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path


class VectorStoreBackend(ABC):
    """RAG 侧向量库统一接口；所有返回的 ``Document`` 应至少保留 ``page_content`` 与原始 ``metadata``。"""

    @property
    @abstractmethod
    def embedding(self) -> Embeddings:
        """返回嵌入模型实例，供 ``embed_queries_for_retrieval`` 做 text_type=query 批量嵌入。"""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """离线索引写入；调用方已做 MD5 文件级去重与切分。"""

    @abstractmethod
    def search_by_vectors(
        self,
        query_vectors: list[list[float]],
        n_results: int,
    ) -> list[Document]:
        """
        多向量批量召回；返回扁平 ``list[Document]``（多查询结果顺序拼接），
        由上层 ``dedupe_*`` 去重/截断。
        """

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> list[Document]:
        """文本单查询召回，作为向量批量召回失败时的兜底路径。"""

    @abstractmethod
    def dump_all(self, *, max_docs: int = 100_000) -> list[Document]:
        """把全量文档拉出来供 ``BM25Retriever.from_documents`` 建索引；仅在 ``hybrid_bm25_enabled`` 时会用到。"""


# ---------------------------------------------------------------------------
# Chroma 实现（默认）
# ---------------------------------------------------------------------------


class ChromaBackend(VectorStoreBackend):
    """基于 langchain-chroma + chromadb 本地持久化的实现。"""

    def __init__(self, embedding: Embeddings, *, collection_name: str, persist_directory: str):
        from langchain_chroma import Chroma  # 延迟导入，和原有直接 import 行为对齐

        self._embedding = embedding
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

    @property
    def embedding(self) -> Embeddings:
        return self._embedding

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._store.add_documents(documents)

    def search_by_vectors(
        self,
        query_vectors: list[list[float]],
        n_results: int,
    ) -> list[Document]:
        if not query_vectors:
            return []
        # Chroma 原生批量接口；返回的 documents/metadatas 是「每路一条」的嵌套列表
        col = self._store._collection  # noqa: SLF001 - 仅在后端内部使用
        raw = col.query(query_embeddings=query_vectors, n_results=n_results)
        return _chroma_flatten(raw)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def dump_all(self, *, max_docs: int = 100_000) -> list[Document]:
        col = self._store._collection  # noqa: SLF001
        batch = col.get(include=["documents", "metadatas"])
        docs_raw = batch.get("documents") or []
        metas = batch.get("metadatas") or [{} for _ in docs_raw]
        if max_docs > 0 and len(docs_raw) > max_docs:
            docs_raw = docs_raw[:max_docs]
            metas = metas[:max_docs]
        return [
            Document(page_content=str(t or ""), metadata=dict(m or {}))
            for t, m in zip(docs_raw, metas)
        ]


def _chroma_flatten(raw: dict[str, Any]) -> list[Document]:
    """解析 Chroma ``collection.query`` 的嵌套结果为扁平 ``list[Document]``。"""
    out: list[Document] = []
    docs_batch = raw.get("documents") or []
    metas_batch = raw.get("metadatas") or []
    for qi in range(len(docs_batch)):
        doc_row = docs_batch[qi] or []
        meta_row = metas_batch[qi] if qi < len(metas_batch) else []
        for j, content in enumerate(doc_row):
            if content is None:
                continue
            md = meta_row[j] if j < len(meta_row) else {}
            out.append(Document(page_content=str(content), metadata=dict(md or {})))
    return out


# ---------------------------------------------------------------------------
# Milvus 实现（可选，需安装 langchain-milvus + pymilvus）
# ---------------------------------------------------------------------------


class MilvusBackend(VectorStoreBackend):
    """
    基于 langchain-milvus 的 Milvus 实现。

    - ``milvus_uri`` 支持远程地址（``http://host:19530``、``grpc://...``）与 Milvus Lite 本地文件路径（相对仓库根解析）。
    - 动态 metadata 通过 ``enable_dynamic_field=True`` 写入，避免为每个 metadata key 定义 schema。
    - 索引/collection 首次写入时由 langchain-milvus 自动创建；要手动建可在 Milvus 侧预先执行。
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        collection_name: str,
        uri: str,
        token: str | None,
        consistency_level: str | None,
        text_field: str,
        vector_field: str,
        primary_field: str,
        drop_old: bool,
    ):
        try:
            from langchain_milvus import Milvus  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "切换到 vector_backend=milvus 需要安装 langchain-milvus 与 pymilvus："
                " pip install 'langchain-milvus' 'pymilvus'；原始错误: " + str(e)
            ) from e

        self._embedding = embedding

        connection_args: dict[str, Any] = {"uri": uri}
        if token:
            connection_args["token"] = token

        kwargs: dict[str, Any] = {
            "embedding_function": embedding,
            "collection_name": collection_name,
            "connection_args": connection_args,
            "text_field": text_field,
            "vector_field": vector_field,
            "primary_field": primary_field,
            "auto_id": True,
            "enable_dynamic_field": True,
            "drop_old": drop_old,
        }
        if consistency_level:
            kwargs["consistency_level"] = consistency_level

        self._store = Milvus(**kwargs)
        logger.info(
            "[RAG] Milvus 已连接 uri=%s collection=%s consistency=%s",
            uri,
            collection_name,
            consistency_level or "<default>",
        )

    @property
    def embedding(self) -> Embeddings:
        return self._embedding

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._store.add_documents(documents)

    def search_by_vectors(
        self,
        query_vectors: list[list[float]],
        n_results: int,
    ) -> list[Document]:
        if not query_vectors:
            return []
        # langchain-milvus 没有直接的「多向量一次发」接口，循环等价语义即可；
        # 每条 query_vector 取 n_results，由上层去重截断池
        out: list[Document] = []
        for vec in query_vectors:
            try:
                docs = self._store.similarity_search_by_vector(vec, k=n_results)
            except Exception as e:  # noqa: BLE001
                logger.warning("[RAG] Milvus similarity_search_by_vector 失败: %s", e)
                continue
            out.extend(docs)
        return out

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def dump_all(self, *, max_docs: int = 100_000) -> list[Document]:
        """
        BM25 全量文本导出。Milvus 对全表扫描没有 Chroma 那样便宜的 ``col.get``，
        这里走 ``pymilvus`` 的 ``query(expr=..., output_fields=["*"])``，并加上 limit 上限。
        在 ``hybrid_bm25_enabled=false`` 时不会触发；开启后建议配合合理的 ``max_docs``。
        """
        try:
            col = getattr(self._store, "col", None)
            if col is None:
                logger.warning("[RAG] Milvus 集合尚未建立，BM25 dump 返回空")
                return []
            # enable_dynamic_field 下动态字段放在 "$meta" 里，pymilvus 的 query 默认会展开
            rows = col.query(
                expr="",  # 空表达式：取全表
                output_fields=["*"],
                limit=max_docs,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[RAG] Milvus dump_all 失败，BM25 跳过: %s", e)
            return []

        text_field = getattr(self._store, "_text_field", None) or "text"
        vector_field = getattr(self._store, "_vector_field", None) or "vector"
        primary_field = getattr(self._store, "_primary_field", None) or "pk"
        reserved = {text_field, vector_field, primary_field}

        out: list[Document] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            content = row.get(text_field)
            if content is None:
                continue
            meta = {k: v for k, v in row.items() if k not in reserved}
            out.append(Document(page_content=str(content), metadata=meta))
        return out


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def _resolve_milvus_uri(raw: str) -> str:
    """远程 uri（带协议）原样返回；裸路径视为 Milvus Lite 本地文件，按仓库根解析。"""
    raw = (raw or "").strip()
    if not raw:
        return "./milvus_rag.db"
    low = raw.lower()
    if "://" in low:
        return raw
    # 相对路径按仓库根解析（与 Chroma persist_directory 语义一致）
    if os.path.isabs(raw):
        return raw
    return resolve_repo_path(raw)


def make_rag_vector_backend(
    embedding: Embeddings,
    *,
    collection_name: str,
    persist_directory: str,
) -> VectorStoreBackend:
    """
    根据 ``chroma_conf["vector_backend"]`` 构造后端实例：

    - ``chroma``（默认）：传入 ``collection_name`` 与 ``persist_directory``；
    - ``milvus``：读取 ``milvus_uri`` / ``milvus_token`` / ``milvus_collection`` / ``milvus_consistency_level``
      等配置；若库未安装会抛 ``RuntimeError``，调用方自行兜底或失败快。
    """
    backend_name = (chroma_conf.get("vector_backend") or "chroma").strip().lower()

    if backend_name == "chroma":
        logger.info(
            "[RAG] 使用 Chroma 向量后端 collection=%s persist_dir=%s",
            collection_name,
            persist_directory,
        )
        return ChromaBackend(
            embedding,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    if backend_name == "milvus":
        uri = _resolve_milvus_uri(str(chroma_conf.get("milvus_uri") or ""))
        token = (chroma_conf.get("milvus_token") or "").strip() or None
        coll = (chroma_conf.get("milvus_collection") or collection_name).strip() or collection_name
        consistency = (chroma_conf.get("milvus_consistency_level") or "").strip() or None
        text_field = (chroma_conf.get("milvus_text_field") or "text").strip() or "text"
        vector_field = (chroma_conf.get("milvus_vector_field") or "vector").strip() or "vector"
        primary_field = (chroma_conf.get("milvus_primary_field") or "pk").strip() or "pk"
        drop_old = bool(chroma_conf.get("milvus_drop_old_on_index", False))
        return MilvusBackend(
            embedding,
            collection_name=coll,
            uri=uri,
            token=token,
            consistency_level=consistency,
            text_field=text_field,
            vector_field=vector_field,
            primary_field=primary_field,
            drop_old=drop_old,
        )

    logger.warning("[RAG] 未知 vector_backend=%s，回退到 Chroma", backend_name)
    return ChromaBackend(
        embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
