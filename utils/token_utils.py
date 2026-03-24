"""
Agent 主模型调用前的输入 token 估算（成本监控）。

使用 tiktoken cl100k_base 对拼接文本做近似计数，与 Qwen/DashScope 账单可能略有偏差，仅供预算参考。
RAG 检索、记忆提取等其它链路不在此统计。
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


def build_agent_llm_input_text_for_token_estimate(state: Any, runtime: Any) -> str:
    """
    与本次 Agent 调用大模型时实际注入内容对齐的文本：system（含记忆）+ messages。

    state: AgentState；runtime: Runtime（需含 context）。
    """
    ctx = getattr(runtime, "context", None) or {}
    is_report = ctx.get("report", False)
    base = load_report_prompts() if is_report else load_system_prompts()
    memory = (ctx.get("memory") or "").strip()
    if memory:
        system = format_memory_system_prompt(memory, base)
    else:
        system = base

    msg_list = state.get("messages") or []
    blocks = [f"[SYSTEM]\n{system}"]
    for i, m in enumerate(msg_list):
        m_type = type(m).__name__
        body = _message_to_estimation_text(m)
        blocks.append(f"--- message[{i}] {m_type} ---\n{body}")
    return "\n\n".join(blocks)


def count_agent_llm_input_tokens(state: Any, runtime: Any) -> int:
    """本次 Agent 调用大模型前的输入 token 估算。"""
    return count_tokens_text(build_agent_llm_input_text_for_token_estimate(state, runtime))
