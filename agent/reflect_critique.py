"""主回答流式结束后：无 ground_truth 的草稿审核；解析/调用失败时放行（fail-open）。"""
from __future__ import annotations

import json
import re
import time

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from model.factory import chat_model
from utils.config_utils import agent_conf, api_conf, rag_conf
from utils.latency_trace import trace_id_or_dash
from utils.log_utils import logger
from utils.prompt_utils import load_reflect_critique_prompts

_PLACEHOLDER_Q = "<<<USER_QUESTION>>>"
_PLACEHOLDER_DRAFT = "<<<DRAFT_ANSWER>>>"

_reflection_llm = None
_reflection_llm_name: str | None = None


class _CritiqueSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: float
    reason: str = ""


def _get_reflection_llm():
    global _reflection_llm, _reflection_llm_name
    name = (agent_conf.get("reflection_model") or "").strip()
    if not name:
        return chat_model
    if _reflection_llm is not None and _reflection_llm_name == name:
        return _reflection_llm
    from langchain_community.chat_models import ChatTongyi

    mk: dict = {}
    if rag_conf.get("chat_temperature") is not None:
        mk["temperature"] = float(rag_conf["chat_temperature"])
    if rag_conf.get("chat_max_tokens") is not None:
        mk["max_tokens"] = int(rag_conf["chat_max_tokens"])

    _reflection_llm_name = name
    _reflection_llm = ChatTongyi(
        dashscope_api_key=api_conf["dashscope_api_key"],
        model=name,
        model_kwargs=mk,
    )
    return _reflection_llm


def _normalize_llm_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _parse_critique_json(text: str) -> tuple[float, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = raw.strip()
    lb, rb = raw.find("{"), raw.rfind("}")
    if lb >= 0 and rb >= lb:
        raw = raw[lb : rb + 1]
    try:
        obj = _CritiqueSchema.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        return None
    s = float(obj.score)
    s = max(0.0, min(1.0, s))
    return s, (obj.reason or "").strip()


def reflect_critique_score(user_question: str, draft_answer: str) -> tuple[float, str]:
    """
    返回 (score, reason)。score 越高越好。
    调用或解析失败时返回 (1.0, "")，避免误触发修正轮。
    """
    template = load_reflect_critique_prompts()
    body = template.replace(_PLACEHOLDER_Q, (user_question or "").strip()).replace(
        _PLACEHOLDER_DRAFT, draft_answer or ""
    )
    try:
        llm = _get_reflection_llm()
        t0 = time.perf_counter()
        out = llm.invoke([HumanMessage(content=body)])
        logger.info(
            "[latency] trace=%s phase=reflection_critique_llm wall_s=%.4f",
            trace_id_or_dash(),
            time.perf_counter() - t0,
        )
        content = _normalize_llm_text(getattr(out, "content", out))
        parsed = _parse_critique_json(content)
        if parsed is None:
            logger.warning(
                "[reflection] 无法解析审核 JSON，放行 | raw=%s",
                content[:300],
            )
            return 1.0, ""
        return parsed
    except Exception as e:
        logger.warning("[reflection] 审核调用失败，放行: %s", e)
        return 1.0, ""
