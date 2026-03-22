"""
RAG 检索管线：可选查询改写、可选 BM25 混合、合并去重、BGE 精排、可选分数拒答。
配置见 config/chroma.yml。
"""
from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from utils.config_utils import chroma_conf
from utils.log_utils import logger

if TYPE_CHECKING:
    from langchain_chroma import Chroma

from model.factory import chat_model


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


def rewrite_queries_for_retrieval(user_query: str) -> list[str]:
    """在原问题基础上生成若干检索用语，提高召回。"""
    max_v = int(chroma_conf.get("query_rewrite_max_variants", 2))
    if max_v <= 0:
        return [user_query.strip()]

    prompt = f"""你是检索查询扩展助手。下面是一条用户问题，请再写出 {max_v} 个**不同表述**的短查询句，用于在同一知识库中做向量检索。
要求：只输出短句本身，每行一条，不要编号、不要解释、不要重复原句。

用户问题：
{user_query.strip()}
"""
    msg = HumanMessage(content=prompt)
    out = chat_model.invoke([msg])
    text = (out.content or "").strip()
    lines = []
    for line in text.splitlines():
        line = line.strip().strip("•-*0123456789.、)）")
        if line and len(line) > 2:
            lines.append(line)
    seen = {user_query.strip()}
    variants: list[str] = [user_query.strip()]
    for line in lines:
        if line not in seen:
            seen.add(line)
            variants.append(line)
        if len(variants) >= max_v + 1:
            break
    logger.info("[RAG] 查询改写: %s 条变体 | %s", len(variants), variants)
    return variants


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

    device = str(chroma_conf.get("rerank_device", "cpu"))
    model_kwargs: dict = {"device": device}
    if chroma_conf.get("rerank_trust_remote_code"):
        model_kwargs["trust_remote_code"] = True

    cache_key = f"{model_name!r}|{device}|{model_kwargs.get('trust_remote_code')}"
    if _cross_encoder_cache and _cross_encoder_cache[0] == cache_key:
        return _cross_encoder_cache[1]

    cross = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
    _cross_encoder_cache = (cache_key, cross)
    return cross


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

    queries = rewrite_queries_for_retrieval(query) if rewrite_on else [query.strip()]

    pool: list[Document] = []
    for q in queries:
        pool.extend(vector_store.similarity_search(q, k=fetch_k))

    if hybrid_on and bm25_retriever is not None:
        try:
            extra = bm25_retriever.invoke(query)
            if isinstance(extra, list):
                pool.extend(extra[:fetch_k])
        except Exception as e:
            logger.warning("[RAG] BM25 检索失败，仅用向量: %s", e)

    pool = dedupe_documents(pool)[:merge_cap]

    rerank_on = bool(chroma_conf.get("rerank_enabled", False))
    if not pool:
        return []

    if not rerank_on:
        return pool[:final_k]

    try:
        cross = get_cached_cross_encoder()
    except ImportError as e:
        raise ImportError("BGE 精排需: pip install sentence-transformers langchain-classic") from e

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
            return []

    return top_docs


REFUSAL_MESSAGE = "当前知识库中未检索到与您问题足够相关的可靠资料，暂无法基于资料作答。请尝试换一种问法或联系人工客服。"
