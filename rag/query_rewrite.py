"""
RAG 查询改写：独立小模型 + LRU 缓存；无运行中事件循环时用 astream/ainvoke，否则同步 stream/invoke。
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, AsyncIterator, Iterator

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

from utils.config_utils import api_conf, chroma_conf
from utils.log_utils import logger

_rewrite_llm: ChatTongyi | None = None
_cache: OrderedDict[str, tuple[str, ...]] | None = None
_cache_max: int = 0


def _cache_key(query: str, max_v: int) -> str:
    q = (query or "").strip()
    return f"{max_v}\n{q}"


def _get_cache() -> OrderedDict[str, tuple[str, ...]] | None:
    global _cache, _cache_max
    max_e = int(chroma_conf.get("query_rewrite_cache_max_entries", 0) or 0)
    if max_e <= 0:
        _cache = None
        return None
    if _cache is None or _cache_max != max_e:
        _cache = OrderedDict()
        _cache_max = max_e
    return _cache


def _get_rewrite_llm() -> ChatTongyi:
    global _rewrite_llm
    if _rewrite_llm is not None:
        return _rewrite_llm
    name = (chroma_conf.get("query_rewrite_model") or "qwen-turbo").strip()
    max_tok = int(chroma_conf.get("query_rewrite_max_tokens", 256))
    temp = float(chroma_conf.get("query_rewrite_temperature", 0))
    _rewrite_llm = ChatTongyi(
        dashscope_api_key=api_conf["dashscope_api_key"],
        model=name,
        model_kwargs={
            "temperature": temp,
            "max_tokens": max_tok,
        },
    )
    logger.info("[RAG] 查询改写模型: %s max_tokens=%s temperature=%s", name, max_tok, temp)
    return _rewrite_llm


def _chunk_text_piece(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(str(b["text"]))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content or "")


def _accumulate_sync_stream(stream: Iterator[Any]) -> str:
    buf: list[str] = []
    for chunk in stream:
        buf.append(_chunk_text_piece(getattr(chunk, "content", None)))
    return "".join(buf).strip()


async def _accumulate_async_stream(stream: AsyncIterator[Any]) -> str:
    buf: list[str] = []
    async for chunk in stream:
        buf.append(_chunk_text_piece(getattr(chunk, "content", None)))
    return "".join(buf).strip()


def _parse_variants(user_query: str, text: str, max_v: int) -> list[str]:
    lines: list[str] = []
    for line in (text or "").splitlines():
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
    return variants


def _rewrite_prompt(user_query: str, max_v: int) -> HumanMessage:
    body = f"""你是检索查询扩展助手。下面是一条用户问题，请再写出 {max_v} 个**不同表述**的短查询句，用于在同一知识库中做向量检索。
要求：只输出短句本身，每行一条，不要编号、不要解释、不要重复原句。

用户问题：
{user_query.strip()}
"""
    return HumanMessage(content=body)


async def _ainvoke_rewrite(user_query: str, max_v: int) -> list[str]:
    llm = _get_rewrite_llm()
    msg = _rewrite_prompt(user_query, max_v)
    use_stream = bool(chroma_conf.get("query_rewrite_stream", True))
    if use_stream:
        text = await _accumulate_async_stream(llm.astream([msg]))
    else:
        out = await llm.ainvoke([msg])
        text = _chunk_text_piece(getattr(out, "content", None)).strip()
    return _parse_variants(user_query, text, max_v)


def _invoke_rewrite(user_query: str, max_v: int) -> list[str]:
    """同步调用；在已有事件循环时（如部分异步框架）使用。"""
    llm = _get_rewrite_llm()
    msg = _rewrite_prompt(user_query, max_v)
    use_stream = bool(chroma_conf.get("query_rewrite_stream", True))
    if use_stream:
        text = _accumulate_sync_stream(llm.stream([msg]))
    else:
        out = llm.invoke([msg])
        text = _chunk_text_piece(getattr(out, "content", None)).strip()
    return _parse_variants(user_query, text, max_v)


def run_query_rewrite(user_query: str, max_v: int) -> list[str]:
    """
    执行改写：无运行中的 asyncio loop 时用 asyncio.run(ainvoke)，否则 invoke。
    DashScope 侧是否真异步取决于 LangChain 实现；小模型 + 低 max_tokens 已显著降耗。
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if in_loop:
        logger.debug("[RAG] 查询改写：检测到运行中事件循环，使用同步 stream/invoke")
        return _invoke_rewrite(user_query, max_v)
    return asyncio.run(_ainvoke_rewrite(user_query, max_v))


def rewrite_queries_cached(user_query: str, max_v: int) -> list[str]:
    """带 LRU 缓存的改写（key = max_v + 原文）。"""
    q = (user_query or "").strip()
    if max_v <= 0:
        return [q]

    c = _get_cache()
    key = _cache_key(q, max_v)
    if c is not None and key in c:
        c.move_to_end(key)
        variants = list(c[key])
        logger.info("[RAG] 查询改写命中缓存: %s 条变体 | %s", len(variants), variants)
        return variants

    variants = run_query_rewrite(q, max_v)
    logger.info("[RAG] 查询改写: %s 条变体 | %s", len(variants), variants)

    if c is not None:
        c[key] = tuple(variants)
        c.move_to_end(key)
        while len(c) > _cache_max:
            c.popitem(last=False)

    return variants
