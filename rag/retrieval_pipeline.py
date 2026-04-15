"""
RAG 检索管线：可选查询改写、可选 BM25 与向量 Ensemble（加权 RRF）、合并去重、
百炼 qwen3-rerank 在线精排（DashScope）、可选分数拒答。
配置见 config/chroma.yml。
"""
from __future__ import annotations

import asyncio  # 用于 asyncio.run(ensemble.ainvoke) 并行跑向量+BM25 两路子检索器
import hashlib
import time
from typing import TYPE_CHECKING, Any

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # 检索器回调类型（LangChain 要求签名）
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever  # 自定义「向量多句」检索器需继承它
from pydantic import ConfigDict  # 允许 vector_store 等非标准字段放进 Pydantic 模型
from rag.dashscope_rerank import DEFAULT_RERANK_URL, dashscope_text_rerank
from rag.parent_store import get_parent_store
from rag.query_embedding_cache import embed_queries_for_retrieval
from rag.query_rewrite import rewrite_queries_cached
from utils.config_utils import api_conf, chroma_conf
from utils.log_utils import log_timing, logger

if TYPE_CHECKING:
    from langchain_chroma import Chroma

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


def dedupe_preserve_order_cap(documents: list[Document], cap: int) -> list[Document]:
    """按 _doc_key 去重，保持输入顺序，最多保留 cap 条（用于 RRF 融合后的池截断）。"""
    # 上限非正：直接返回空，避免无意义遍历
    if cap <= 0:
        return []
    # 已经出现过的「来源+正文指纹」，用于去重
    seen: set[str] = set()
    # 去重后按原顺序收集的文档列表
    out: list[Document] = []
    # 按 RRF 排好的顺序依次看每条文档（靠前的分数更高）
    for d in documents:
        # 与 dedupe_documents 相同的键：同一段内容、同一 source 视为重复
        k = _doc_key(d)
        # 这条已经选过了，跳过（保留第一次出现 = 保留更高 RRF 排名那条）
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
        # 凑够 merge_max_pool 条就停，后面不再进精排池
        if len(out) >= cap:
            break
    return out


class _FixedQueriesChromaRetriever(BaseRetriever):
    """对固定查询列表做 Chroma 批量向量召回；invoke 传入的 query 可忽略（Ensemble 仍传用户原句）。"""

    # Chroma 客户端不是 Pydantic 标准类型，必须允许任意类型字段
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any  # 已连好的 Chroma 向量库实例
    queries: list[str]  # 已算好的检索句列表（含原问 + 改写句）
    fetch_k: int  # 每句从向量库取几条候选

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # LangChain 会传入 query，但这里故意只用 self.queries（改写结果在进 Ensemble 前就定好了）
        return _vector_search_chroma_batch(self.vector_store, self.queries, self.fetch_k)


def _invoke_ensemble_hybrid(ensemble: Any, query: str) -> list[Document]:
    """优先 ainvoke + asyncio.run 以并行子检索器；已在事件循环内则退回顺序 invoke。"""
    # 探测当前线程是否已在 asyncio 事件循环里（例如在 async Web 框架中）
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的循环：可以安全地用 asyncio.run 包一层
        try:
            # ainvoke 内部会对多个子 retriever 并发 gather，向量与 BM25 同时跑
            return asyncio.run(ensemble.ainvoke(query))
        except Exception as e:
            # 异步路径失败（环境限制等）：退回同步 invoke，两路顺序执行
            logger.warning("[RAG] Ensemble ainvoke 失败，退回 invoke: %s", e)
            return ensemble.invoke(query)
    # 已在事件循环内：不能再 asyncio.run，否则嵌套会报错，只能同步顺序 invoke
    logger.debug("[RAG] Ensemble：检测到运行中事件循环，使用同步 invoke")
    return ensemble.invoke(query)


def _build_hybrid_ensemble_retriever(
        multi_query_vector_retriever: _FixedQueriesChromaRetriever,
        bm25_retriever: Any,
) -> Any:
    """向量多句一路 + BM25 一路；可选 chroma 两路权重，不配则 Ensemble 默认等权 RRF。"""
    vector_branch_weight = chroma_conf.get("ensemble_vector_weight")
    bm25_branch_weight = chroma_conf.get("ensemble_bm25_weight")
    ensemble_kwargs: dict[str, Any] = {
        "retrievers": [multi_query_vector_retriever, bm25_retriever],
    }
    if vector_branch_weight is not None or bm25_branch_weight is not None:
        ensemble_kwargs["weights"] = [
            float(0.5 if vector_branch_weight is None else vector_branch_weight),
            float(0.5 if bm25_branch_weight is None else bm25_branch_weight),
        ]
    return EnsembleRetriever(**ensemble_kwargs)


def _recall_candidate_documents(
        multi_query_vector_retriever: _FixedQueriesChromaRetriever,
        normalized_user_query: str,
        bm25_retriever: Any | None,
        hybrid_bm25_config_enabled: bool,
) -> tuple[list[Document], bool]:
    """
    粗召回：可选「向量 + BM25」Ensemble（RRF），否则仅向量多句 batch。
    返回 (候选文档列表, 是否已做 RRF 融合)；后者决定后续用哪种去重截断策略。
    """
    hybrid_retrieval_active = hybrid_bm25_config_enabled and bm25_retriever is not None
    if not hybrid_retrieval_active:
        documents = multi_query_vector_retriever.invoke(normalized_user_query)
        return documents, False

    try:
        hybrid_ensemble = _build_hybrid_ensemble_retriever(
            multi_query_vector_retriever,
            bm25_retriever,
        )
        documents = _invoke_ensemble_hybrid(hybrid_ensemble, normalized_user_query)
        return documents, True
    except Exception as exc:
        logger.warning("[RAG] Ensemble 混合检索失败，回退仅向量: %s", exc)
        documents = multi_query_vector_retriever.invoke(normalized_user_query)
        return documents, False


def expand_parent_documents(documents: list[Document]) -> list[Document]:
    """
    父子块：子块命中后，仅根据 metadata 中的 parent_id 从 SQLite 映射表取父块正文。
    无 parent_id、未配置映射表、或表中无该 id 时，保留子块正文；同一父块多子块命中只输出一条父块。
    """
    if not documents:
        return []
    store = get_parent_store()
    parent_ids_ordered: list[str] = []
    for d in documents:
        pid = (d.metadata or {}).get("parent_id")
        if pid:
            parent_ids_ordered.append(str(pid))
    by_id: dict[str, str] = {}
    if store and parent_ids_ordered:
        by_id = store.get_many(list(dict.fromkeys(parent_ids_ordered)))

    out: list[Document] = []
    seen_parent: set[str] = set()
    for d in documents:
        meta = dict(d.metadata or {})
        pid = meta.get("parent_id")
        if not pid:
            out.append(d)
            continue
        pid_s = str(pid)
        ptext = by_id.get(pid_s)
        if ptext:
            if pid_s in seen_parent:
                continue
            seen_parent.add(pid_s)
            meta.pop("parent_content", None)
            meta["expanded_from"] = "parent_child"
            out.append(Document(page_content=str(ptext), metadata=meta))
            continue
        if store:
            logger.debug(
                "[RAG] parent_id=%s 在映射表中无记录，使用子块正文",
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


def _dashscope_rerank_pairs(
        query: str,
        expanded_documents: list[Document],
) -> list[tuple[Document, float]]:
    """百炼 text-rerank：返回 (Document, relevance_score)，顺序与 API results 一致（已按分数降序）。"""
    texts = [str(d.page_content or "") for d in expanded_documents]
    model = str(chroma_conf.get("rerank_model") or "qwen3-rerank")
    base_url = str(chroma_conf.get("rerank_api_url") or DEFAULT_RERANK_URL).strip()
    top_n = min(500, len(texts))
    return_documents = bool(chroma_conf.get("rerank_return_documents", True))
    instruct = chroma_conf.get("rerank_instruct")
    timeout_s = float(chroma_conf.get("rerank_timeout_seconds", 60.0))
    api_key = str(api_conf.get("dashscope_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 dashscope_api_key（config/api.yml 或 DASHSCOPE_API_KEY）")

    results = dashscope_text_rerank(
        query=query,
        documents=texts,
        api_key=api_key,
        model=model,
        base_url=base_url,
        top_n=top_n,
        return_documents=return_documents,
        instruct=str(instruct).strip() if instruct else None,
        timeout_s=timeout_s,
    )
    n = len(expanded_documents)
    out: list[tuple[Document, float]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            continue
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= n:
            continue
        score = item.get("relevance_score")
        if score is None:
            continue
        out.append((expanded_documents[i], float(score)))
    return out


def retrieve_documents(
        vector_store: Chroma,
        query: str,
        *,
        bm25_retriever: Any | None,
) -> list[Document]:
    """
    端到端检索：粗召回 → 池截断 → 父子块展开（Small-to-Big）→ 可选百炼 qwen3-rerank 精排 → 取 top-k；
    可按配置因分数阈值拒答（返回空列表）。精排在展开之后，使打分与最终进上下文的正文一致。
    各段耗时：grep [timing] scope=rag_retrieve，用相邻两条 wall= 时间戳对减。
    """
    log_timing("rag_retrieve", "retrieve_enter")
    # --- 最终要交给生成模型的片段条数、向量侧每句拉取条数（与 chroma.yml 中 k / fetch_k 对应）---
    context_document_count = int(chroma_conf["k"])
    vector_fetch_count = int(chroma_conf.get("fetch_k", max(context_document_count, 20)))
    if vector_fetch_count < context_document_count:
        logger.warning(
            "[RAG] fetch_k(%s) < k(%s)，已把 fetch_k 调整为 k",
            vector_fetch_count,
            context_document_count,
        )
        vector_fetch_count = context_document_count

    query_rewrite_config_enabled = bool(chroma_conf.get("query_rewrite_enabled", False))
    hybrid_bm25_config_enabled = bool(chroma_conf.get("hybrid_bm25_enabled", False))
    merge_pool_max_size = int(chroma_conf.get("merge_max_pool", 60))

    # --- 查询改写：得到用于向量检索的短句列表（首条通常为原问）---
    if query_rewrite_config_enabled:
        retrieval_query_texts = rewrite_queries_for_retrieval(query)
    else:
        retrieval_query_texts = [query.strip()]
    log_timing("rag_retrieve", "retrieval_queries_ready")

    multi_query_vector_retriever = _FixedQueriesChromaRetriever(
        vector_store=vector_store,
        queries=retrieval_query_texts,
        fetch_k=vector_fetch_count,
    )
    normalized_user_query = query.strip()

    # --- 粗召回：可选向量+BM25 RRF；再按策略去重并限制进入精排池的规模 ---
    candidate_documents, recall_used_reciprocal_rank_fusion = _recall_candidate_documents(
        multi_query_vector_retriever,
        normalized_user_query,
        bm25_retriever,
        hybrid_bm25_config_enabled,
    )
    if recall_used_reciprocal_rank_fusion:
        # RRF 已按融合分排序：在去重时保留更靠前的同键文档
        candidate_documents = dedupe_preserve_order_cap(candidate_documents, merge_pool_max_size)
    else:
        # 纯向量：先去重再截取前 merge_pool_max_size 条
        candidate_documents = dedupe_documents(candidate_documents)[:merge_pool_max_size]
    log_timing("rag_retrieve", "recall_and_truncate_done")

    rerank_enabled = bool(chroma_conf.get("rerank_enabled", False))

    if not candidate_documents:
        log_timing(
            "rag_retrieve",
            "summary",
            rewrite_on=query_rewrite_config_enabled,
            rerank_on=rerank_enabled,
            pool_len=0,
            out_len=0,
        )
        return []

    # --- Small-to-Big：先按 parent_id 展成父块（与最终进上下文的段落一致），再在线精排 ---
    # 若先对子块 rerank 再换父块，分数与送入 LLM 的正文不对齐；业界常见做法是「最终 passage 上重排」。
    expanded_documents = expand_parent_documents(candidate_documents)
    log_timing("rag_retrieve", "parent_expand_done")

    if not rerank_enabled:
        context_documents = expanded_documents[:context_document_count]
        log_timing(
            "rag_retrieve",
            "summary",
            rewrite_on=query_rewrite_config_enabled,
            rerank_on=False,
            pool_len=len(expanded_documents),
            out_len=len(context_documents),
        )
        return context_documents

    # --- 百炼 qwen3-rerank：对展开后的 passage 打分，再取 top 或按阈值过滤 ---
    log_timing("rag_retrieve", "dashscope_rerank_start")
    try:
        documents_sorted_by_rerank_score = _dashscope_rerank_pairs(
            normalized_user_query,
            expanded_documents,
        )
    except Exception as exc:
        logger.warning(
            "[RAG] DashScope rerank 失败，回退为粗排顺序截断: %s",
            exc,
        )
        documents_sorted_by_rerank_score = [
            (document, 1.0 - index * 1e-9)
            for index, document in enumerate(expanded_documents)
        ]
    if not documents_sorted_by_rerank_score:
        logger.warning("[RAG] DashScope rerank 返回空结果，回退为粗排顺序截断")
        documents_sorted_by_rerank_score = [
            (document, 1.0 - index * 1e-9)
            for index, document in enumerate(expanded_documents)
        ]

    # 配置 rerank_refuse_min_score：低于阈值的片段丢弃；若全无则拒答（空列表）。
    rerank_refuse_minimum_score = chroma_conf.get("rerank_refuse_min_score")
    if rerank_refuse_minimum_score is None:
        reranked_top_documents = [
            document
            for document, _score in documents_sorted_by_rerank_score[:context_document_count]
        ]
    else:
        minimum_rerank_score_threshold = float(rerank_refuse_minimum_score)
        documents_above_threshold: list[Document] = []
        for document, score in documents_sorted_by_rerank_score:
            if float(score) < minimum_rerank_score_threshold:
                continue
            documents_above_threshold.append(document)
            if len(documents_above_threshold) >= context_document_count:
                break
        reranked_top_documents = documents_above_threshold
        if not reranked_top_documents:
            best_score = (
                float(documents_sorted_by_rerank_score[0][1])
                if documents_sorted_by_rerank_score
                else float("-inf")
            )
            logger.warning(
                "[RAG] 拒答：rerank 后无片段通过阈值 best=%.4f thr=%.4f",
                best_score,
                minimum_rerank_score_threshold,
            )
            log_timing("rag_retrieve", "dashscope_rerank_done")
            log_timing(
                "rag_retrieve",
                "summary",
                rewrite_on=query_rewrite_config_enabled,
                rerank_on=True,
                pool_len=len(expanded_documents),
                out_len=0,
            )
            return []

    log_timing("rag_retrieve", "dashscope_rerank_done")
    context_documents = reranked_top_documents
    log_timing(
        "rag_retrieve",
        "summary",
        rewrite_on=query_rewrite_config_enabled,
        rerank_on=True,
        pool_len=len(expanded_documents),
        out_len=len(context_documents),
    )
    return context_documents


REFUSAL_MESSAGE = "当前知识库中未检索到与您问题足够相关的可靠资料，暂无法基于资料作答。请尝试换一种问法或联系人工客服。"
