import json
import os
import urllib.error
import urllib.request

import streamlit as st

from agent.checkpoint_react_agent import make_checkpoint_thread_id
from memory.history_store import get_history_store
from utils.log_utils import logger, set_session_id
from utils.sse_format import SSEDecoder

_SSE_PORT = int(os.environ.get("RAW_AGENT_SSE_PORT", "8765"))
if os.environ.get("RAW_AGENT_EMBED_SSE", "1").strip().lower() not in ("0", "false", "no"):
    try:
        from api.sse_embed import ensure_embedded_sse_server

        ensure_embedded_sse_server(port=_SSE_PORT)
    except Exception as e:
        logger.warning("[app] 内嵌 SSE 服务未启动（可设置 RAW_AGENT_EMBED_SSE=0 并自行 uvicorn api.sse_chat:app）: %s", e)


def _sse_chat_base_url() -> str:
    base = (os.environ.get("RAW_AGENT_SSE_BASE") or "").strip().rstrip("/")
    if base:
        return base
    if os.environ.get("RAW_AGENT_EMBED_SSE", "1").strip().lower() not in ("0", "false", "no"):
        try:
            from api.sse_embed import ensure_embedded_sse_server

            ensure_embedded_sse_server(port=_SSE_PORT)
        except Exception:
            pass
    return f"http://127.0.0.1:{_SSE_PORT}"


def _consume_agent_sse(
        url: str,
        body: bytes,
        live_placeholder,
        *,
        error_hint_url: str,
) -> dict:
    """解析 chat/stream 或 chat/resume 的 SSE，返回 assistant 正文与 HITL 状态。"""
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        },
    )
    decoder = SSEDecoder()
    merged: list[str] = []
    display_final = ""
    hitl_waiting = False
    thread_from_done: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                for event, data in decoder.feed(chunk):
                    if event == "chunk":
                        obj = json.loads(data)
                        d = obj.get("delta") or ""
                        if d:
                            merged.append(d)
                            live_placeholder.markdown("".join(merged))
                    elif event == "done":
                        obj = json.loads(data)
                        display_final = (obj.get("last_turn_display_assistant_text") or "").strip()
                        hitl_waiting = bool(obj.get("hitl_waiting"))
                        thread_from_done = obj.get("thread_id")
                    elif event == "error":
                        obj = json.loads(data)
                        raise RuntimeError(obj.get("error") or "SSE error")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SSE HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 SSE 服务 {error_hint_url}。若未使用内嵌服务，请先运行: "
            f"uvicorn api.sse_chat:app --port {_SSE_PORT}，并设置 RAW_AGENT_SSE_BASE。"
        ) from e
    out = display_final or "".join(merged).strip()
    if out:
        live_placeholder.markdown(out)
    return {
        "assistant_text": out,
        "hitl_waiting": hitl_waiting,
        "thread_id": thread_from_done,
    }


def _run_chat_sse(
        messages_for_agent: list[dict],
        *,
        user_id: str | None,
        thread_id: str,
        session_tag: str | None,
        live_placeholder,
) -> dict:
    """POST /v1/chat/stream；返回 assistant_text、hitl_waiting、thread_id。"""
    base = _sse_chat_base_url()
    url = f"{base}/v1/chat/stream"
    payload = {
        "messages": messages_for_agent,
        "user_id": user_id,
        "thread_id": thread_id,
        "session_tag": session_tag,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _consume_agent_sse(url, body, live_placeholder, error_hint_url=base)


def _run_chat_resume(
        resume_text: str,
        *,
        user_id: str | None,
        thread_id: str,
        live_placeholder,
) -> dict:
    """POST /v1/chat/resume，同 checkpoint 线程续跑人机回环。"""
    base = _sse_chat_base_url()
    url = f"{base}/v1/chat/resume"
    payload = {
        "resume": resume_text,
        "user_id": user_id,
        "thread_id": thread_id,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _consume_agent_sse(url, body, live_placeholder, error_hint_url=base)


# 标题
st.title("RAW智能助手")
st.divider()

# 每个浏览器会话独立的 session_state → 对话隔离
if "_session_tag" not in st.session_state:
    import uuid

    st.session_state["_session_tag"] = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

st.session_state.setdefault("_history_loaded_user", "")
st.session_state.setdefault("_prev_user_id", "")
history_store = get_history_store()


def _history_row(m: dict) -> dict:
    row = {"role": m["role"], "content": m["content"]}
    if m.get("hitl_clarification"):
        row["hitl_clarification"] = True
    return row


with st.sidebar:
    st.caption("本会话的对话仅保存在当前浏览器标签页，与其他访问者隔离。")
    st.caption(
        f"SSE：`{_sse_chat_base_url()}`（/v1/chat/stream、人机回环 /v1/chat/resume）；"
        "内嵌开关 RAW_AGENT_EMBED_SSE，端口 RAW_AGENT_SSE_PORT。"
    )
    user_id = (
            st.text_input(
                "当前用户名",
                placeholder="请输入用户名",
                key="user_id",
            )
            or ""
    ).strip()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("切换", help="加载该用户的历史对话"):
            if user_id:
                st.session_state["_prev_user_id"] = user_id
                st.session_state["_history_loaded_user"] = ""
                loaded = history_store.get_messages(user_id)
                st.session_state["chat_messages"] = [_history_row(m) for m in loaded]
                st.session_state["_history_loaded_user"] = user_id
            else:
                st.session_state["chat_messages"] = []
            st.rerun()
    with col2:
        if st.button("清空本轮对话"):
            st.session_state["chat_messages"] = []
            st.session_state.pop("_hitl_waiting", None)
            st.session_state.pop("_hitl_thread_id", None)
            # 同时重置 checkpoint thread，避免新对话继承旧图状态
            import uuid

            st.session_state["_session_tag"] = str(uuid.uuid4())
            st.rerun()

# 长期记忆：按 user_id 加载完整历史（切换用户且消息为空时）
user_id = (st.session_state.get("user_id") or "").strip()
if user_id and st.session_state["chat_messages"] == [] and st.session_state["_history_loaded_user"] != user_id:
    loaded = history_store.get_messages(user_id)
    if loaded:
        st.session_state["chat_messages"] = [_history_row(m) for m in loaded]
    st.session_state["_history_loaded_user"] = user_id

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        if message.get("hitl_clarification"):
            st.caption("补充说明（人机回环）")
        st.write(message["content"])

if st.session_state.get("_hitl_waiting"):
    st.info("助手正在等待您对上一问的补充说明；下一条输入会继续同一轮对话（不是新话题）。")

prompt = st.chat_input()

if prompt:
    user_msg: dict = {"role": "user", "content": prompt}
    if st.session_state.get("_hitl_waiting"):
        user_msg["hitl_clarification"] = True
    with st.chat_message("user"):
        if user_msg.get("hitl_clarification"):
            st.caption("补充说明（人机回环）")
        st.write(prompt)
    st.session_state["chat_messages"].append(user_msg)

    # 长期记忆：持久化完整历史（仅当有 user_id）
    if user_id:
        hist_kw = {}
        if user_msg.get("hitl_clarification"):
            hist_kw["hitl_clarification"] = True
        history_store.append_message(user_id, "user", prompt, **hist_kw)

    set_session_id(user_id)

    with st.spinner("agent思考中..."):
        messages_for_agent = list(st.session_state["chat_messages"])
        tid = make_checkpoint_thread_id(user_id or None, st.session_state.get("_session_tag") or None)
        resume_tid = (st.session_state.get("_hitl_thread_id") or tid).strip()
        assistant_slot = st.chat_message("assistant").empty()
        try:
            if st.session_state.get("_hitl_waiting"):
                chat_out = _run_chat_resume(
                    prompt,
                    user_id=user_id or None,
                    thread_id=resume_tid,
                    live_placeholder=assistant_slot,
                )
            else:
                chat_out = _run_chat_sse(
                    messages_for_agent,
                    user_id=user_id or None,
                    thread_id=tid,
                    session_tag=st.session_state.get("_session_tag") or None,
                    live_placeholder=assistant_slot,
                )
            assistant_text = chat_out.get("assistant_text") or ""
            if chat_out.get("hitl_waiting"):
                st.session_state["_hitl_waiting"] = True
                st.session_state["_hitl_thread_id"] = (chat_out.get("thread_id") or resume_tid).strip()
            else:
                st.session_state["_hitl_waiting"] = False
                st.session_state.pop("_hitl_thread_id", None)
        except Exception as e:
            assistant_text = f"[错误] {e!s}"
            assistant_slot.markdown(assistant_text)
            st.session_state["_hitl_waiting"] = False
            st.session_state.pop("_hitl_thread_id", None)
            logger.exception("[app] SSE 对话失败")

    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": assistant_text}
    )
    # 长期记忆：持久化 assistant 回复（记忆抽取在服务端 execute_stream 结束并入队，见 memory/memory_queue.py）
    if user_id and assistant_text:
        history_store.append_message(user_id, "assistant", assistant_text)
    st.rerun()
