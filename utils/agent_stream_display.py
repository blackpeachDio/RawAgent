"""
Agent 流式输出：前台可先展示推理/思考过程，结束后仅保留面向用户的正文。

- DashScope（Qwen）等会把推理放在 additional_kwargs["reasoning_content"]，正文在 content。
- 部分模型会把 think / reasoning 等标签嵌在 content 字符串中。
- LangChain AIMessage.text 仅聚合 type=text 的块，不含 reasoning 块。
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

# Qwen / 常见推理标签（content 内嵌时）
_THINK_PATTERNS = (
    re.compile(r"(?is)<think\b[^>]*>.*?</think>"),
    re.compile(r"(?is)<reasoning\b[^>]*>.*?</reasoning>"),
)


def strip_thinking_markers(text: str) -> str:
    """去掉 content 中嵌套的思考标签块，并压缩多余空行。"""
    if not text:
        return ""
    s = text
    for pat in _THINK_PATTERNS:
        s = pat.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _reasoning_from_kwargs(msg: AIMessage) -> str:
    ak = msg.additional_kwargs
    if not ak:
        return ""
    r = ak.get("reasoning_content")
    return r if isinstance(r, str) else ""


def _blocks_stream_visible(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text") or "")
            elif t in ("reasoning", "thinking"):
                parts.append(block.get("reasoning") or block.get("thinking") or "")
    return "".join(parts)


def assistant_stream_visible_text(msg: AIMessage) -> str:
    """
    用于流式 delta 计算：与用户可见过程一致（含 reasoning_content + 正文）。
    """
    rc = _reasoning_from_kwargs(msg)
    c = msg.content
    if isinstance(c, str):
        return rc + c
    if isinstance(c, list):
        return rc + _blocks_stream_visible(c)
    return rc


def assistant_final_display_text(msg: AIMessage) -> str:
    """
    结束后写入会话历史 / 持久化：仅保留面向用户的正文，不含推理过程。
    """
    if isinstance(msg.content, str):
        return strip_thinking_markers(msg.content.strip())
    # 列表块：仅用 LangChain 聚合的 text 块
    return strip_thinking_markers(str(msg.text).strip())
