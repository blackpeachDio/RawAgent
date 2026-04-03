"""
将 LangGraph `stream_mode="values"` 的状态流解析为 UI 可用事件（工具调用、工具结果、文本增量）。

与具体 Agent 构建方式解耦：调用方传入已编译图的 `.stream(...)` 迭代器即可。
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from utils.log_utils import logger

_TOOL_RESULT_MAX_CHARS = 1200
_TOOL_ARGS_PREVIEW_MAX_CHARS = 800


def _last_aimessage_index(msgs: list[Any]) -> int | None:
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], AIMessage):
            return i
    return None


def _aimessage_text(msg: AIMessage) -> str:
    """AIMessage.content 转纯文本（含多模态列表中的 text 块）。"""
    c = msg.content
    if c is None:
        return ""
    if isinstance(c, list):
        parts: list[str] = []
        for x in c:
            if isinstance(x, dict) and x.get("type") == "text":
                parts.append(str(x.get("text", "")))
            elif isinstance(x, str):
                parts.append(x)
        return "".join(parts).strip()
    return str(c).strip()


def _tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("name") or "")
    return str(getattr(tc, "name", "") or "")


def _tool_call_args(tc: Any) -> Any:
    if isinstance(tc, dict):
        return tc.get("args")
    return getattr(tc, "args", None)


def _format_tool_args(args: Any) -> str:
    if args is None:
        return ""
    try:
        if isinstance(args, dict):
            return json.dumps(args, ensure_ascii=False)
        return str(args)
    except Exception:
        return str(args)


def _toolmessage_content(m: ToolMessage) -> str:
    c = m.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for x in c:
            if isinstance(x, dict) and "text" in x:
                parts.append(str(x["text"]))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(c)


def iter_stream_events_from_values_stream(
        stream_iter: Iterator[dict[str, Any]],
        *,
        recursion_limit: int,
) -> Iterator[dict[str, Any]]:
    """
    消费 `graph.stream(..., stream_mode="values")` 的 chunk 序列，产出结构化事件。

    Yields:
        {"type": "tool_call", "name": str, "args_preview": str, "id": str}
        {"type": "tool_result", "name": str, "content_preview": str, "tool_call_id": str}
        {"type": "text_delta", "content": str}
        {"type": "error", "content": str}  # 仅递归上限等
    """
    seen_tool_call_ids: set[str] = set()
    prev_len = 0
    prev_ai_index: int | None = None
    prev_assistant_text = ""

    try:
        for chunk in stream_iter:
            raw_msgs = chunk.get("messages") or []

            if len(raw_msgs) > prev_len:
                for m in raw_msgs[prev_len:]:
                    if isinstance(m, ToolMessage):
                        preview = _toolmessage_content(m)
                        if len(preview) > _TOOL_RESULT_MAX_CHARS:
                            preview = preview[:_TOOL_RESULT_MAX_CHARS] + "…"
                        yield {
                            "type": "tool_result",
                            "name": m.name or "",
                            "content_preview": preview,
                            "tool_call_id": m.tool_call_id or "",
                        }
                prev_len = len(raw_msgs)

            ai_idx = _last_aimessage_index(raw_msgs)
            if ai_idx is not None:
                last_ai = cast(AIMessage, raw_msgs[ai_idx])
                for tc in last_ai.tool_calls or []:
                    tid = _tool_call_id(tc)
                    if tid and tid in seen_tool_call_ids:
                        continue
                    if tid:
                        seen_tool_call_ids.add(tid)
                    ap = _format_tool_args(_tool_call_args(tc))
                    if len(ap) > _TOOL_ARGS_PREVIEW_MAX_CHARS:
                        ap = ap[:_TOOL_ARGS_PREVIEW_MAX_CHARS] + "…"
                    yield {
                        "type": "tool_call",
                        "name": _tool_call_name(tc),
                        "args_preview": ap,
                        "id": tid,
                    }

                if ai_idx != prev_ai_index:
                    prev_assistant_text = ""
                    prev_ai_index = ai_idx

                text = _aimessage_text(last_ai)
                if not text:
                    continue
                if len(text) > len(prev_assistant_text) and text.startswith(
                        prev_assistant_text
                ):
                    delta = text[len(prev_assistant_text):]
                    prev_assistant_text = text
                    if delta:
                        yield {"type": "text_delta", "content": delta}
                elif text != prev_assistant_text:
                    prev_assistant_text = text
                    yield {
                        "type": "text_delta",
                        "content": text + ("\n" if not text.endswith("\n") else ""),
                    }
    except GraphRecursionError as e:
        logger.warning(
            "[values_stream_events] 已达图执行步数上限 recursion_limit=%s: %s",
            recursion_limit,
            e,
        )
        yield {
            "type": "error",
            "content": (
                "\n\n[提示] 本轮智能体推理步数已达上限，已停止。"
                "若问题较复杂，可拆成更短的问题分步提问，或在配置中适当调大 agent_recursion_limit。"
            ),
        }
