"""
Agent 对话 SSE 接口：POST /v1/chat/stream，响应 text/event-stream。

请求 JSON：
  messages: [{role, content}, ...]  # 与 execute_stream 相同，最后一条须为 user
  user_id, thread_id, session_tag  # 可选，与 CheckpointReactAgent.execute_stream 一致

事件：
  event: chunk  data: {"delta": "<增量字符串>"}
  event: done   data: {"ok": true, "last_turn_display_assistant_text": "..."}
  event: error  data: {"ok": false, "error": "..."}

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

from agent.checkpoint_react_agent import CheckpointReactAgent
from utils.log_utils import logger
from utils.sse_format import format_sse_event

_agent: CheckpointReactAgent | None = None
_stream_lock = threading.Lock()


def get_agent() -> CheckpointReactAgent:
    global _agent
    if _agent is None:
        _agent = CheckpointReactAgent()
    return _agent


def _chat_stream_generator(payload: dict[str, Any]):
    messages = payload.get("messages") or []
    user_id = payload.get("user_id")
    thread_id = payload.get("thread_id")
    session_tag = payload.get("session_tag")
    agent = get_agent()
    with _stream_lock:
        try:
            for delta in agent.execute_stream(
                    messages,
                    user_id=user_id,
                    thread_id=thread_id,
                    session_tag=session_tag,
            ):
                line = json.dumps({"delta": delta}, ensure_ascii=False)
                yield format_sse_event(line, event="chunk")
            done = {
                "ok": True,
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


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "service": "raw-agent-sse"})


def build_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/v1/chat/stream", endpoint=chat_stream, methods=["POST"]),
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
