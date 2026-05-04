"""
Agent 主模型调用前的输入 token 估算（成本监控）。

使用 DashScope ``get_tokenizer`` 对拼接文本做 token 计数，与 Qwen 线上计费的分词口径一致；
仅做"进入 LLM 前的输入体积"估算，RAG 检索、记忆提取等其它链路不在此统计。

优先使用 ``ModelRequest``（``@wrap_model_call``）估算：含解析后的 system、非 system 的 messages、
tools 的 schema 文本、可选的 tool_choice / response_format；比仅用 state+runtime 更接近真实请求。
厂商对 tools 的实际序列化可能与本地拼接略有差异，属于可接受误差。
"""
from __future__ import annotations

import json
from typing import Any

from dashscope import get_tokenizer

from utils.config_utils import rag_conf
from utils.log_utils import logger


_TOKENIZER_CACHE: dict[str, Any] = {}
_FALLBACK_MODEL = "qwen-turbo"


def _default_model_name() -> str:
    """主 Agent 使用的模型名；与 model.factory.ChatModelFactory 保持一致。"""
    name = (rag_conf.get("chat_model_name") or "").strip()
    return name or _FALLBACK_MODEL


def _get_tokenizer(model: str | None = None) -> Any:
    """按模型名缓存 tokenizer；首次加载失败时回退到 qwen-turbo，仍失败则返回 None。"""
    name = (model or _default_model_name()).strip() or _FALLBACK_MODEL
    if name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[name]
    try:
        tok = get_tokenizer(name)
    except Exception as e:
        logger.warning("[token] 无法加载 dashscope tokenizer=%s: %s", name, e)
        if name != _FALLBACK_MODEL:
            try:
                tok = get_tokenizer(_FALLBACK_MODEL)
                logger.info("[token] 已回退到 dashscope tokenizer=%s", _FALLBACK_MODEL)
            except Exception as e2:
                logger.warning("[token] 回退 tokenizer 也加载失败: %s", e2)
                tok = None
        else:
            tok = None
    _TOKENIZER_CACHE[name] = tok
    return tok


def count_tokens_text(text: str, *, model: str | None = None) -> int:
    """
    对纯文本做 token 估算。

    - 正常路径：使用 DashScope ``get_tokenizer`` 对应的 Qwen 分词器。
    - 兜底：tokenizer 不可用时用字符长度做最粗略估算，保证调用方不因计数失败而抛错。
    """
    if not text:
        return 0
    tok = _get_tokenizer(model)
    if tok is None:
        return len(text)
    try:
        return len(tok.encode(text))
    except Exception as e:
        logger.warning("[token] encode 失败，退回字符长度估算: %s", e)
        return len(text)


def _message_to_estimation_text(m: Any) -> str:
    """将单条 LangChain message 转为可计数的文本（含 tool_calls 等）。"""
    parts: list[str] = []
    content = getattr(m, "content", None)
    parts.append(str(content or "").strip())
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        try:
            parts.append("[tool_calls]\n" + json.dumps(tool_calls, ensure_ascii=False))
        except Exception:
            parts.append("[tool_calls]\n" + str(tool_calls))
    name = getattr(m, "name", None)
    if name:
        parts.append(f"[name]={name}")
    return "\n".join(p for p in parts if p)


def _system_text_from_model_request(request: Any) -> str:
    sm = getattr(request, "system_message", None)
    if sm is not None:
        c = getattr(sm, "content", None)
        if c is not None:
            return str(c)
    sp = getattr(request, "system_prompt", None)
    if sp:
        return str(sp)
    return ""


def _tool_to_estimation_text(tool: Any) -> str:
    """将单个 tool 转为可计数的文本（名称、描述、参数 schema）。"""
    if isinstance(tool, dict):
        try:
            return json.dumps(tool, ensure_ascii=False)
        except Exception:
            return str(tool)
    name = getattr(tool, "name", None) or "unknown"
    desc = (getattr(tool, "description", None) or "").strip()
    params: Any = {}
    try:
        get_schema = getattr(tool, "get_input_schema", None)
        if callable(get_schema):
            sch = get_schema()
            params = sch.model_json_schema() if hasattr(sch, "model_json_schema") else str(sch)
    except Exception:
        try:
            params = getattr(tool, "args", {}) or {}
        except Exception:
            params = {}
    try:
        return json.dumps({"name": name, "description": desc, "parameters": params}, ensure_ascii=False)
    except Exception:
        return f"{name}\n{desc}\n{params}"


def build_agent_llm_input_text_for_model_request(request: Any) -> str:
    """
    与单次 ``ModelRequest`` 对齐的输入文本：system_message / system_prompt、messages、tools、
    可选 tool_choice / response_format（粗算进体积）。

    用于 ``@wrap_model_call``，比 state+runtime 重放更接近真实请求。
    """
    blocks: list[str] = []
    system = _system_text_from_model_request(request)
    if system:
        blocks.append(f"[SYSTEM]\n{system}")

    msg_list = getattr(request, "messages", None) or []
    for i, m in enumerate(msg_list):
        m_type = type(m).__name__
        body = _message_to_estimation_text(m)
        blocks.append(f"--- message[{i}] {m_type} ---\n{body}")

    tools = getattr(request, "tools", None) or []
    if tools:
        tool_lines = [_tool_to_estimation_text(t) for t in tools]
        blocks.append("[TOOLS]\n" + "\n".join(tool_lines))

    tc = getattr(request, "tool_choice", None)
    if tc is not None:
        try:
            blocks.append("[TOOL_CHOICE]\n" + json.dumps(tc, default=str, ensure_ascii=False))
        except Exception:
            blocks.append(f"[TOOL_CHOICE]\n{tc!s}")

    rf = getattr(request, "response_format", None)
    if rf is not None:
        try:
            blocks.append("[RESPONSE_FORMAT]\n" + json.dumps(rf, default=str, ensure_ascii=False))
        except Exception:
            blocks.append(f"[RESPONSE_FORMAT]\n{rf!s}")

    return "\n\n".join(blocks)


def count_agent_llm_input_tokens_from_model_request(request: Any) -> dict[str, int]:
    """
    基于 ``ModelRequest`` 的输入 token 分段估算。

    返回键：
        - ``system``：system prompt 段 token 数
        - ``messages``：所有非 system 对话消息（含 tool_calls）token 数
        - ``tools``：工具 schema 段 token 数（无工具时为 0）
        - ``extras``：tool_choice / response_format 等附加段 token 数
        - ``total``：上述各项之和（与整体文本单独计数可能差 1~2，属于分段边界误差）
    """
    model_name = _default_model_name()

    system = _system_text_from_model_request(request)
    system_tokens = count_tokens_text(f"[SYSTEM]\n{system}", model=model_name) if system else 0

    msg_list = getattr(request, "messages", None) or []
    msg_blocks: list[str] = []
    for i, m in enumerate(msg_list):
        body = _message_to_estimation_text(m)
        msg_blocks.append(f"--- message[{i}] {type(m).__name__} ---\n{body}")
    messages_tokens = (
        count_tokens_text("\n\n".join(msg_blocks), model=model_name) if msg_blocks else 0
    )

    tools = getattr(request, "tools", None) or []
    tools_tokens = 0
    if tools:
        tools_tokens = count_tokens_text(
            "[TOOLS]\n" + "\n".join(_tool_to_estimation_text(t) for t in tools),
            model=model_name,
        )

    extras_parts: list[str] = []
    tc = getattr(request, "tool_choice", None)
    if tc is not None:
        try:
            extras_parts.append("[TOOL_CHOICE]\n" + json.dumps(tc, default=str, ensure_ascii=False))
        except Exception:
            extras_parts.append(f"[TOOL_CHOICE]\n{tc!s}")
    rf = getattr(request, "response_format", None)
    if rf is not None:
        try:
            extras_parts.append("[RESPONSE_FORMAT]\n" + json.dumps(rf, default=str, ensure_ascii=False))
        except Exception:
            extras_parts.append(f"[RESPONSE_FORMAT]\n{rf!s}")
    extras_tokens = (
        count_tokens_text("\n\n".join(extras_parts), model=model_name) if extras_parts else 0
    )

    total = system_tokens + messages_tokens + tools_tokens + extras_tokens
    return {
        "system": system_tokens,
        "messages": messages_tokens,
        "tools": tools_tokens,
        "extras": extras_tokens,
        "total": total,
    }
