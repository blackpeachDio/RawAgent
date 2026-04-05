"""按配置与用户当前问句，决定是否注入事实记忆 / 向量记忆。"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from model.factory import default_turbo_chat_model_name, make_turbo_chat_model
from utils.log_utils import logger


class MemoryInjectDecision(BaseModel):
    """LLM 分类输出：与提示词中的 JSON Schema 一致，供 Pydantic 校验。"""

    model_config = ConfigDict(extra="forbid")

    inject_factual: bool
    inject_vector: bool


# auto + keywords 或未配置关键词时使用的默认子串（任一命中则注入对应类型）
_DEFAULT_FACTUAL_KEYWORDS = [
    "用户画像",
    "偏好",
    "爱好",
    "我叫",
    "个人信息",
    "我住",
    "记住我",
    "我的情况",
]
_DEFAULT_VECTOR_KEYWORDS = [
    "上次",
    "之前",
    "记得",
    "说过",
    "历史",
    "摘要",
    "对话",
    "事件",
    "曾经",
    "以前",
    "提过",
]

_CLASSIFIER_PLACEHOLDER = "<<<USER_QUERY>>>"
_classifier_prompt_cached: str | None = None
_classifier_schema_for_prompt: str | None = None
_classifier_llm: Any = None
_classifier_llm_name: str | None = None


def _classifier_json_schema_text() -> str:
    """与 MemoryInjectDecision 同步的 Schema，拼进提示词以约束模型。"""
    global _classifier_schema_for_prompt
    if _classifier_schema_for_prompt is None:
        _classifier_schema_for_prompt = json.dumps(
            MemoryInjectDecision.model_json_schema(),
            ensure_ascii=False,
        )
    return _classifier_schema_for_prompt


def _keyword_list(conf: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    if key not in conf or conf[key] is None:
        return defaults
    return list(conf[key])


def _flags_from_keywords(query: str, conf: dict[str, Any]) -> tuple[bool, bool]:
    factual_kw = _keyword_list(
        conf, "memory_inject_factual_keywords", _DEFAULT_FACTUAL_KEYWORDS
    )
    vector_kw = _keyword_list(
        conf, "memory_inject_vector_keywords", _DEFAULT_VECTOR_KEYWORDS
    )
    q = (query or "").strip()
    inject_f = bool(q and any(k in q for k in factual_kw))
    inject_v = bool(q and any(k in q for k in vector_kw))
    return inject_f, inject_v


def _classifier_prompt_text(query: str) -> str:
    global _classifier_prompt_cached
    if _classifier_prompt_cached is None:
        from utils.prompt_utils import load_memory_inject_classifier_prompt

        _classifier_prompt_cached = load_memory_inject_classifier_prompt()
    body = _classifier_prompt_cached.replace(_CLASSIFIER_PLACEHOLDER, (query or "").strip())
    return (
        body
        + "\n\n【JSON Schema】你的回复必须且仅能是满足该 Schema 的一个 JSON 对象：\n"
        + _classifier_json_schema_text()
    )


def _get_classifier_llm(conf: dict[str, Any]) -> Any:
    global _classifier_llm, _classifier_llm_name
    name = (conf.get("memory_inject_classifier_model") or "").strip() or default_turbo_chat_model_name()
    if _classifier_llm is not None and _classifier_llm_name == name:
        return _classifier_llm

    _classifier_llm_name = name
    _classifier_llm = make_turbo_chat_model(model=name, max_tokens=256, temperature=0)
    return _classifier_llm


def _parse_llm_classifier_output(text: str) -> tuple[bool, bool] | None:
    """
    约定模型仅输出一个 JSON 对象：strip 后直接解析并做 Pydantic 校验。
    若模型夹带前后缀导致非纯 JSON，解析失败并回退关键词。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        decision = MemoryInjectDecision.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.debug("[memory_inject] JSON 校验失败: %s | raw=%s", e, raw[:200])
        return None
    return decision.inject_factual, decision.inject_vector


def _flags_from_llm(query: str, conf: dict[str, Any]) -> tuple[bool, bool] | None:
    q = (query or "").strip()
    if not q:
        return False, False
    try:
        llm = _get_classifier_llm(conf)
        prompt = _classifier_prompt_text(q)
        msg = HumanMessage(content=prompt)
        out = llm.invoke([msg])
        content = out.content if hasattr(out, "content") else str(out)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            content = "".join(parts)
        parsed = _parse_llm_classifier_output(str(content))
        if parsed is None:
            logger.warning(
                "[memory_inject] LLM 分类结果无法解析，将回退关键词 | raw=%s",
                str(content)[:200],
            )
        return parsed
    except Exception as e:
        logger.warning("[memory_inject] LLM 分类调用失败，将回退关键词: %s", e)
        return None


def memory_inject_flags(query: str, conf: dict[str, Any]) -> tuple[bool, bool]:
    """
    返回 (是否注入事实记忆, 是否注入向量记忆)。

    memory_inject_mode:
      - always: 两者均 True
      - never: 两者均 False
      - auto: 由 memory_inject_auto_strategy 决定：
        - llm（默认）：轻量模型输出 JSON；失败则回退关键词
        - keywords：子串匹配（与旧逻辑一致）
    """
    mode = (conf.get("memory_inject_mode") or "always").strip().lower()
    q = (query or "").strip()

    if mode == "never":
        return False, False
    if mode == "always":
        return True, True
    if mode != "auto":
        return True, True

    strategy = (conf.get("memory_inject_auto_strategy") or "llm").strip().lower()

    if strategy == "keywords":
        return _flags_from_keywords(query, conf)

    if strategy != "llm":
        logger.warning(
            "[memory_inject] 未知 memory_inject_auto_strategy=%s，按 llm 处理",
            strategy,
        )

    llm_result = _flags_from_llm(query, conf)
    if llm_result is not None:
        return llm_result
    return _flags_from_keywords(query, conf)
