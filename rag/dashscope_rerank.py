"""
阿里云百炼（DashScope）文本重排序 HTTP 调用。

与 curl 一致：POST .../text-rerank/text-rerank，Bearer DASHSCOPE_API_KEY。
响应见官方文档：output.results[] 含 index、relevance_score（0~1，越大越相关）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from utils.log_utils import logger

DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


def dashscope_text_rerank(
        *,
        query: str,
        documents: list[str],
        api_key: str,
        model: str,
        base_url: str,
        top_n: int,
        return_documents: bool,
        instruct: str | None,
        timeout_s: float,
) -> list[dict[str, Any]]:
    """
    调用百炼 text-rerank，返回 output.results（已按 relevance_score 降序）。

    每项至少含 index（对应 input.documents 下标）、relevance_score。
    """
    if not documents:
        return []
    q = (query or "").strip()
    if not q:
        raise ValueError("rerank query 不能为空")

    payload: dict[str, Any] = {
        "model": model,
        "input": {
            "query": q,
            "documents": list(documents),
        },
        "parameters": {
            "return_documents": bool(return_documents),
            "top_n": int(top_n),
        },
    }
    if instruct and str(instruct).strip():
        payload["parameters"]["instruct"] = str(instruct).strip()

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        str(base_url).strip(),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        logger.warning(
            "[RAG] DashScope rerank HTTP %s: %s",
            getattr(e, "code", "?"),
            err_body[:500] or str(e),
        )
        raise RuntimeError(f"DashScope rerank HTTP {getattr(e, 'code', '?')}") from e
    except urllib.error.URLError as e:
        logger.warning("[RAG] DashScope rerank 网络错误: %s", e)
        raise

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("[RAG] DashScope rerank 响应非 JSON: %s", raw[:300])
        raise RuntimeError("DashScope rerank 响应解析失败") from e

    if data.get("code"):
        msg = data.get("message") or str(data.get("code"))
        logger.warning("[RAG] DashScope rerank 业务错误: %s", msg)
        raise RuntimeError(msg)

    output = data.get("output") or {}
    results = output.get("results")
    if not isinstance(results, list):
        logger.warning("[RAG] DashScope rerank 缺少 output.results: %s", raw[:400])
        raise RuntimeError("DashScope rerank 响应格式异常")

    return results
