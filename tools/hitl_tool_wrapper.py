"""
将任意 StructuredTool 包装为「调用前先人机回环」的工具。

流程：模型给出入参 → interrupt（展示 proposed_args）→ 用户 resume：
  - approve / 同意：按原参执行
  - modify / 修改：合并覆盖部分或全部入参后再执行（经 schema 校验）
  - reject / 拒绝：不调用工具，返回固定说明字符串

resume 载荷（推荐 JSON 字符串，也可 dict）示例::

    {"action": "approve"}
    {"action": "reject", "reason": "不需要查天气"}
    {"action": "modify", "args": {"city": "上海"}}

亦支持简短纯文本：同意 / approve / ok；拒绝 / reject。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt


_HITL_DESC_APPEND = (
    "\n\n【人机确认】执行前需用户在界面确认：可同意、修改参数或拒绝；"
    "拒绝时不得假定本工具已返回结果。"
)


@dataclass
class _Decision:
    action: Literal["approve", "reject", "modify"]
    args: dict[str, Any] | None = None
    reason: str = ""


def _parse_resume(raw: Any) -> _Decision:
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        raise ValueError("人机回环 resume 为空")

    if isinstance(raw, dict):
        d = dict(raw)
    else:
        s = str(raw).strip()
        low = s.lower()
        if low in ("approve", "ok", "yes", "y", "同意", "确认", "执行"):
            return _Decision("approve")
        if low in ("reject", "no", "n", "拒绝", "取消"):
            return _Decision("reject", reason="用户拒绝")
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析人机回环 resume: {s[:200]}") from e

    action_raw = (d.get("action") or d.get("decision") or "").strip().lower()
    alias = {
        "approve": "approve",
        "同意": "approve",
        "confirm": "approve",
        "modify": "modify",
        "修改": "modify",
        "edit": "modify",
        "reject": "reject",
        "拒绝": "reject",
        "deny": "reject",
    }
    action = alias.get(action_raw, action_raw)
    if action not in ("approve", "reject", "modify"):
        raise ValueError(f"无效 action: {action_raw!r}，须为 approve | modify | reject")

    reason = (d.get("reason") or d.get("message") or "").strip()
    args = d.get("args")
    if args is not None and not isinstance(args, dict):
        raise ValueError("modify 时 args 须为 JSON 对象")

    if action == "modify":
        if not args:
            raise ValueError("modify 时须提供 args 对象（可与 proposed 合并）")
        return _Decision("modify", args=args, reason=reason)

    return _Decision(action, args=None, reason=reason)  # type: ignore[arg-type]


def _append_description(description: str | None) -> str:
    base = (description or "").strip()
    return base + _HITL_DESC_APPEND


def wrap_tool_with_pre_invoke_hitl(
        inner: BaseTool,
        *,
        description_suffix: str | None = None,
) -> StructuredTool:
    """
    传入 LangChain ``BaseTool``（须具备 ``args_schema``），返回包装后的 ``StructuredTool``。

    仅在图已启用 checkpoint、且客户端能对同一 ``thread_id`` 调用 ``/v1/chat/resume``
    注入 resume 时可用（与 ``request_user_clarification`` 相同前提）。

    :param inner: 被包装的工具（如同文件中的 ``get_weather``）
    :param description_suffix: 追加到描述末尾的自定义说明（可选）
    """
    schema = inner.args_schema
    if schema is None:
        raise ValueError("wrap_tool_with_pre_invoke_hitl 需要工具带有 args_schema（显式注解参数）")

    suffix = description_suffix or ""
    description = _append_description(inner.description or "") + suffix

    def _sync(**kwargs: Any) -> str:
        proposed = dict(kwargs)
        payload: dict[str, Any] = {
            "kind": "tool_hitl",
            "questions": (
                f"即将调用工具 `{inner.name}`，拟定参数："
                f"{json.dumps(proposed, ensure_ascii=False)}。"
                f"请确认：同意执行 / 修改参数（resume 传 {{\"action\":\"modify\",\"args\":{{...}}}}）/ 拒绝。"
            ),
            "tool_name": inner.name,
            "proposed_args": proposed,
            "missing_slots": [],
            "reason": "",
        }
        raw = interrupt(payload)
        dec = _parse_resume(raw)

        if dec.action == "reject":
            r = dec.reason or "用户拒绝执行该工具"
            return (
                f"[用户已拒绝执行工具 `{inner.name}`，请勿假定已获得工具输出。] {r}"
            ).strip()

        if dec.action == "modify":
            merged = {**proposed, **(dec.args or {})}
            validated = schema.model_validate(merged)
            args_out = validated.model_dump()
        else:
            validated = schema.model_validate(proposed)
            args_out = validated.model_dump()

        out = inner.invoke(args_out)
        if not isinstance(out, str):
            out = str(out)
        return out

    return StructuredTool.from_function(
        _sync,
        name=inner.name,
        description=description,
        args_schema=schema,
    )
