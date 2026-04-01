"""
Agent 主模型调用前的输入 token 估算（成本监控）。

使用 tiktoken cl100k_base 对拼接文本做近似计数，与 Qwen/DashScope 账单可能略有偏差，仅供预算参考。
RAG 检索、记忆提取等其它链路不在此统计。

优先使用 ``ModelRequest``（``@wrap_model_call``）估算：含解析后的 system、非 system 的 messages、
tools 的 schema 文本；比仅用 state+runtime 更接近真实进模型体积。厂商对 tools 的实际序列化
可能与拼接方式略有差异。
"""
from __future__ import annotations

import json
from typing import Any

import tiktoken

from utils.prompt_utils import format_memory_system_prompt, load_report_prompts, load_system_prompts

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens_text(text: str) -> int:
    """对纯文本做 token 估算。"""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


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


def count_agent_llm_input_tokens_from_model_request(request: Any) -> int:
    """基于 ModelRequest 的输入 token 估算（含 tools 等）。"""
    return count_tokens_text(build_agent_llm_input_text_for_model_request(request))
