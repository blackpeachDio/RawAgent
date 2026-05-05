"""
Agent 对话 SSE 接口：POST /v1/chat/stream，响应 text/event-stream。

请求 JSON（/v1/chat/stream）：
  messages: [{role, content}, ...]  # 与 execute_stream 相同，最后一条须为 user
  user_id, thread_id, session_tag  # 可选，与 CheckpointReactAgent.execute_stream 一致

POST /v1/chat/resume（人机回环续跑）：
  thread_id: str  # 须与挂起时一致
  resume: str     # 用户补充说明（纯文本）
  user_id: str | null  # 可选

事件：
  event: chunk        data: {"delta": "<增量字符串>"}
  event: hitl_pending data: {"thread_id","questions","missing_slots","run_id","reason"}
  event: done         data: {"ok", "hitl_waiting", "thread_id", "last_turn_display_assistant_text", ...}
  event: error        data: {"ok": false, "error": "..."}

运行：uvicorn api.sse_chat:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import json
import threading
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from agent.checkpoint_react_agent import (
    CheckpointReactAgent,
    make_checkpoint_thread_id,
)
from utils.log_utils import logger
from utils.sse_format import format_sse_event

_agent: CheckpointReactAgent | None = None
_stream_lock = threading.Lock()


def get_agent() -> CheckpointReactAgent:
    global _agent
    if _agent is None:
        _agent = CheckpointReactAgent()
    return _agent


def _hitl_pending_payload(agent: CheckpointReactAgent, thread_id: str) -> dict[str, Any]:
    meta = agent._hitl_wait_by_thread.get(thread_id) or {}
    out: dict[str, Any] = {
        "thread_id": thread_id,
        "questions": meta.get("questions") or "",
        "missing_slots": meta.get("missing_slots") or [],
        "run_id": meta.get("run_id"),
        "reason": meta.get("reason") or "",
    }
    for k in ("kind", "tool_name", "proposed_args"):
        if k in meta and meta[k] is not None:
            out[k] = meta[k]
    return out


def _chat_stream_generator(payload: dict[str, Any]):
    messages = payload.get("messages") or []
    user_id = payload.get("user_id")
    thread_id = payload.get("thread_id")
    session_tag = payload.get("session_tag")
    agent = get_agent()
    with _stream_lock:
        try:
            tid = (thread_id or "").strip() or None
            for delta in agent.execute_stream(
                    messages,
                    user_id=user_id,
                    thread_id=thread_id,
                    session_tag=session_tag,
            ):
                line = json.dumps({"delta": delta}, ensure_ascii=False)
                yield format_sse_event(line, event="chunk")
            # thread_id 以 agent 侧解析为准（与 execute_stream 内 tid 一致）
            eff_tid = tid or make_checkpoint_thread_id(user_id, session_tag)
            if eff_tid in agent._hitl_wait_by_thread:
                hp = json.dumps(_hitl_pending_payload(agent, eff_tid), ensure_ascii=False)
                yield format_sse_event(hp, event="hitl_pending")
            done = {
                "ok": True,
                "hitl_waiting": eff_tid in agent._hitl_wait_by_thread,
                "thread_id": eff_tid,
                "last_turn_display_assistant_text": (
                    getattr(agent, "last_turn_display_assistant_text", "") or ""
                ),
            }
            yield format_sse_event(
                json.dumps(done, ensure_ascii=False),
                event="done",
            )
        except Exception as e:
            logger.exception("[sse_chat] stream failed")
            err = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            yield format_sse_event(err, event="error")


def _resume_stream_generator(payload: dict[str, Any]):
    thread_id = (payload.get("thread_id") or "").strip()
    resume = payload.get("resume")
    user_id = payload.get("user_id")
    agent = get_agent()
    with _stream_lock:
        try:
            if not thread_id:
                raise ValueError("thread_id 不能为空")
            for delta in agent.execute_resume_stream(
                    resume_text=str(resume or ""),
                    thread_id=thread_id,
                    user_id=user_id,
            ):
                line = json.dumps({"delta": delta}, ensure_ascii=False)
                yield format_sse_event(line, event="chunk")
            if thread_id in agent._hitl_wait_by_thread:
                hp = json.dumps(_hitl_pending_payload(agent, thread_id), ensure_ascii=False)
                yield format_sse_event(hp, event="hitl_pending")
            done = {
                "ok": True,
                "hitl_waiting": thread_id in agent._hitl_wait_by_thread,
                "thread_id": thread_id,
                "last_turn_display_assistant_text": (
                    getattr(agent, "last_turn_display_assistant_text", "") or ""
                ),
            }
            yield format_sse_event(
                json.dumps(done, ensure_ascii=False),
                event="done",
            )
        except Exception as e:
            logger.exception("[sse_chat] resume failed")
            err = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            yield format_sse_event(err, event="error")


async def chat_stream(request: Request) -> StreamingResponse | JSONResponse:
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    return StreamingResponse(
        _chat_stream_generator(payload),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def chat_resume(request: Request) -> StreamingResponse | JSONResponse:
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    return StreamingResponse(
        _resume_stream_generator(payload),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "service": "raw-agent-sse"})


def build_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/v1/chat/stream", endpoint=chat_stream, methods=["POST"]),
            Route("/v1/chat/resume", endpoint=chat_resume, methods=["POST"]),
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = build_app()

if __name__ == "__main__":
    import os

    import uvicorn

    _port = int(os.environ.get("RAW_AGENT_SSE_PORT", "8765"))
    uvicorn.run("api.sse_chat:app", host="127.0.0.1", port=_port)
