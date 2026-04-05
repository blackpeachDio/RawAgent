"""
RAG 检索管线：可选查询改写、可选 BM25 混合、合并去重、BGE 精排、可选分数拒答。
配置见 config/chroma.yml。
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from rag.parent_store import get_parent_store
from rag.query_embedding_cache import embed_queries_for_retrieval
from rag.query_rewrite import rewrite_queries_cached
from utils.config_utils import chroma_conf
from utils.latency_trace import note_rag_retrieve_breakdown
from utils.log_utils import logger

if TYPE_CHECKING:
    from langchain_chroma import Chroma

def _apply_hf_hub_endpoint() -> None:
    if os.environ.get("HF_ENDPOINT"):
        return
    ep = chroma_conf.get("hf_endpoint") or chroma_conf.get("hf_mirror")
    if ep and str(ep).strip():
        os.environ["HF_ENDPOINT"] = str(ep).strip().rstrip("/")
        logger.info("[RAG] HF_ENDPOINT=%s", os.environ["HF_ENDPOINT"])


def _doc_key(d: Document) -> str:
    src = (d.metadata or {}).get("source") or ""
    h = hashlib.md5((d.page_content or "").encode("utf-8", errors="ignore")).hexdigest()
    return f"{src}::{h}"


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in documents:
        k = _doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


def expand_parent_documents(documents: list[Document]) -> list[Document]:
    """
    父子块索引：命中子块后展开为父块正文。
    优先从 SQLite 映射表按 parent_id 取全文；若无表或查不到，则回退 metadata 中的 parent_content（旧索引兼容）。
    """
    if not documents:
        return []
    store = get_parent_store()
    need_fetch: list[str] = []
    for d in documents:
        m = d.metadata or {}
        pid = m.get("parent_id")
        if not pid or m.get("parent_content"):
            continue
        need_fetch.append(str(pid))
    by_id: dict[str, str] = {}
    if store and need_fetch:
        by_id = store.get_many(list(dict.fromkeys(need_fetch)))

    out: list[Document] = []
    seen_parent: set[str] = set()
    for d in documents:
        meta = dict(d.metadata or {})
        pid = meta.get("parent_id")
        if not pid:
            out.append(d)
            continue
        pid_s = str(pid)
        ptext = meta.get("parent_content")
        if not ptext:
            ptext = by_id.get(pid_s)
        if not ptext and store:
            ptext = store.get(pid_s)
        if ptext:
            if pid_s in seen_parent:
                continue
            seen_parent.add(pid_s)
            meta.pop("parent_content", None)
            meta["expanded_from"] = "parent_child"
            out.append(Document(page_content=str(ptext), metadata=meta))
        else:
            logger.warning(
                "[RAG] 父子展开失败：parent_id=%s 无映射且无 parent_content，保留子块",
                pid_s,
            )
            out.append(d)
    return out


def _chroma_query_result_to_documents(raw: dict[str, Any]) -> list[Document]:
    """解析 Chroma `collection.query` 多路结果（documents/metadatas 为「每路一条」的嵌套列表）。"""
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


def _vector_search_chroma_batch(
        vector_store: Chroma,
        queries: list[str],
        fetch_k: int,
) -> list[Document]:
    """
    向量召回：统一走 collection.query + 检索侧向量缓存。
    DashScope 使用 text_type=query 批量嵌入（避免误用 document 类型）；重复问句命中 LRU 则跳过远程。
    """
    if not queries:
        return []
    emb = getattr(vector_store, "_embedding_function", None)
    if emb is None:
        raise ValueError(
            "向量检索必须配置 embedding_function（见 rag/online_query 等）。"
        )
    try:
        col = vector_store._collection  # noqa: SLF001
        vecs = embed_queries_for_retrieval(emb, queries)
        raw = col.query(query_embeddings=vecs, n_results=fetch_k)
        return _chroma_query_result_to_documents(raw)
    except Exception as e:
        logger.warning("[RAG] Chroma query 失败，回退 similarity_search: %s", e)
        pool: list[Document] = []
        for q in queries:
            pool.extend(vector_store.similarity_search(q, k=fetch_k))
        return pool


def warmup_vector_retrieval(vector_store: Chroma, *, sample_query: str = " ") -> None:
    """预热 Chroma + 查询嵌入链路（DashScope 仍会有一次网络；后续同会话命中缓存）。"""
    if not bool(chroma_conf.get("recall_warmup_on_startup", False)):
        return
    t0 = time.perf_counter()
    try:
        q = (sample_query or " ").strip() or " "
        _vector_search_chroma_batch(vector_store, [q], 1)
        logger.info("[RAG] 向量召回预热完成 wall_s=%.4f", time.perf_counter() - t0)
    except Exception as e:
        logger.warning("[RAG] 向量召回预热失败: %s", e)


def rewrite_queries_for_retrieval(user_query: str) -> list[str]:
    """在原问题基础上生成若干检索用语；独立小模型 + 缓存 + 优先异步 ainvoke（见 rag/query_rewrite.py）。"""
    max_v = int(chroma_conf.get("query_rewrite_max_variants", 2))
    return rewrite_queries_cached(user_query, max_v)


def build_bm25_retriever(vector_store: Chroma):
    from langchain_community.retrievers import BM25Retriever

    col = vector_store._collection  # noqa: SLF001
    batch = col.get(include=["documents", "metadatas"])
    docs_raw = batch.get("documents") or []
    metas = batch.get("metadatas") or [{} for _ in docs_raw]
    if not docs_raw:
        logger.warning("[RAG] BM25：向量库无文本，跳过混合检索")
        return None
    docs = [
        Document(page_content=str(t), metadata=m or {})
        for t, m in zip(docs_raw, metas)
    ]
    fk = int(chroma_conf.get("fetch_k", 20))
    return BM25Retriever.from_documents(docs, k=fk)


_cross_encoder_cache: tuple[str, Any] | None = None


def get_cached_cross_encoder():
    """同一进程内复用 CrossEncoder，避免每次检索重复加载模型。"""
    global _cross_encoder_cache
    from utils.path_utils import get_abs_path

    _apply_hf_hub_endpoint()
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model_raw = chroma_conf.get("rerank_model") or "BAAI/bge-reranker-base"
    if model_raw.startswith("BAAI/") or model_raw.startswith("intfloat/"):
        model_name = model_raw
    elif os.path.isdir(model_raw):
        model_name = model_raw
    else:
        cand = get_abs_path(model_raw)
        model_name = cand if os.path.isdir(cand) else model_raw

    # 设备选择
    requested_device = str(chroma_conf.get("rerank_device", "cpu"))
    device = requested_device
    if requested_device.startswith("cuda"):
        # 部分环境的 torch 未编译 CUDA 时，强行把模型 device 设为 cuda 会触发
        # `AssertionError: Torch not compiled with CUDA enabled`。
        try:
            import torch

            cuda_built = False
            try:
                cuda_built = bool(torch.backends.cuda.is_built)
            except Exception:
                cuda_built = False

            cuda_available = False
            try:
                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = False

            if not cuda_built or not cuda_available:
                logger.warning(
                    "[RAG] rerank_device=%s 但 CUDA不可用（built=%s available=%s），回退到 cpu",
                    requested_device,
                    cuda_built,
                    cuda_available,
                )
                device = "cpu"
        except Exception as e:
            logger.warning(
                "[RAG] 检测 CUDA 是否可用失败：%s，回退到 cpu", str(e)
            )
            device = "cpu"

    model_kwargs: dict = {"device": device}
    if chroma_conf.get("rerank_trust_remote_code"):
        model_kwargs["trust_remote_code"] = True

    cache_key = f"{model_name!r}|{device}|{model_kwargs.get('trust_remote_code')}"
    if _cross_encoder_cache and _cross_encoder_cache[0] == cache_key:
        return _cross_encoder_cache[1]

    cross = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
    _cross_encoder_cache = (cache_key, cross)
    return cross


def preload_rerank_cross_encoder() -> None:
    """
    进程内提前加载 BGE CrossEncoder（与 retrieve_documents 共用缓存）。
    在 rerank_enabled=true 时由 ReactAgent 初始化调用，避免首次 RAG 才加载导致首问卡顿。
    """
    if not bool(chroma_conf.get("rerank_enabled", False)):
        logger.debug("[RAG] rerank_enabled=false，跳过 BGE CrossEncoder 预加载")
        return
    try:
        t0 = time.perf_counter()
        get_cached_cross_encoder()
        logger.info(
            "[RAG] BGE CrossEncoder 启动预加载完成 wall_s=%.3f",
            time.perf_counter() - t0,
        )
    except Exception as e:
        logger.warning(
            "[RAG] BGE CrossEncoder 启动预加载失败，将在首次需要重排时再加载: %s",
            e,
        )


def retrieve_documents(
        vector_store: Chroma,
        query: str,
        *,
        bm25_retriever: Any | None,
) -> list[Document]:
    """
    粗排（多路向量 ± BM25）→ 合并去重 → BGE 精排 → 可选拒答（空列表）。
    """
    final_k = int(chroma_conf["k"])
    fetch_k = int(chroma_conf.get("fetch_k", max(final_k, 20)))
    if fetch_k < final_k:
        logger.warning(
            "[RAG] fetch_k(%s) < k(%s)，已把 fetch_k 调整为 k",
            fetch_k,
            final_k,
        )
        fetch_k = final_k

    rewrite_on = bool(chroma_conf.get("query_rewrite_enabled", False))
    hybrid_on = bool(chroma_conf.get("hybrid_bm25_enabled", False))
    merge_cap = int(chroma_conf.get("merge_max_pool", 60))

    rewrite_s = 0.0
    if rewrite_on:
        t_rw = time.perf_counter()
        queries = rewrite_queries_for_retrieval(query)
        rewrite_s = time.perf_counter() - t_rw
    else:
        queries = [query.strip()]

    pool: list[Document] = []
    do_bm25 = hybrid_on and bm25_retriever is not None

    t_re = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        vec_fut = ex.submit(_vector_search_chroma_batch, vector_store, queries, fetch_k)
        bm25_fut = ex.submit(bm25_retriever.invoke, query) if do_bm25 else None

        pool.extend(vec_fut.result())

        if bm25_fut is not None:
            try:
                extra = bm25_fut.result()
                if isinstance(extra, list):
                    pool.extend(extra[:fetch_k])
            except Exception as e:
                logger.warning("[RAG] BM25 检索失败，仅用向量: %s", e)

    pool = dedupe_documents(pool)[:merge_cap]
    recall_s = time.perf_counter() - t_re

    rerank_on = bool(chroma_conf.get("rerank_enabled", False))
    if not pool:
        note_rag_retrieve_breakdown(
            rewrite_on=rewrite_on,
            rewrite_s=rewrite_s,
            recall_s=recall_s,
            rerank_on=rerank_on,
            rerank_s=0.0,
            expand_s=0.0,
            pool_len=0,
            out_len=0,
        )
        return []

    rerank_s = 0.0
    expand_s = 0.0
    out_len = 0

    if not rerank_on:
        t_exp = time.perf_counter()
        out = expand_parent_documents(pool[:final_k])
        expand_s = time.perf_counter() - t_exp
        out_len = len(out)
        note_rag_retrieve_breakdown(
            rewrite_on=rewrite_on,
            rewrite_s=rewrite_s,
            recall_s=recall_s,
            rerank_on=False,
            rerank_s=0.0,
            expand_s=expand_s,
            pool_len=len(pool),
            out_len=out_len,
        )
        return out

    try:
        cross = get_cached_cross_encoder()
    except ImportError as e:
        raise ImportError("BGE 精排需: pip install sentence-transformers langchain-classic") from e

    t_rr = time.perf_counter()
    pairs = [(query.strip(), d.page_content) for d in pool]
    raw_scores = cross.score(pairs)
    scores = list(raw_scores)
    ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:final_k]]
    top_scores = [s for _, s in ranked[:final_k]]

    min_score = chroma_conf.get("rerank_refuse_min_score")
    if min_score is not None and top_scores:
        thr = float(min_score)
        best = float(top_scores[0])
        if best < thr:
            logger.warning(
                "[RAG] 拒答：精排最高分 %.4f < 阈值 %.4f",
                best,
                thr,
            )
            rerank_s = time.perf_counter() - t_rr
            note_rag_retrieve_breakdown(
                rewrite_on=rewrite_on,
                rewrite_s=rewrite_s,
                recall_s=recall_s,
                rerank_on=True,
                rerank_s=rerank_s,
                expand_s=0.0,
                pool_len=len(pool),
                out_len=0,
            )
            return []

    rerank_s = time.perf_counter() - t_rr
    t_exp = time.perf_counter()
    out = expand_parent_documents(top_docs)
    expand_s = time.perf_counter() - t_exp
    out_len = len(out)
    note_rag_retrieve_breakdown(
        rewrite_on=rewrite_on,
        rewrite_s=rewrite_s,
        recall_s=recall_s,
        rerank_on=True,
        rerank_s=rerank_s,
        expand_s=expand_s,
        pool_len=len(pool),
        out_len=out_len,
    )
    return out


REFUSAL_MESSAGE = "当前知识库中未检索到与您问题足够相关的可靠资料，暂无法基于资料作答。请尝试换一种问法或联系人工客服。"
