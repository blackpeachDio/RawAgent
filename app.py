import time

import streamlit as st

from agent.react_agent import ReactAgent
from memory.history_store import get_history_store
from utils.log_utils import set_session_id

# 标题
st.title("RAW智能助手")
st.divider()

# 每个浏览器会话独立的 session_state → 对话隔离
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

st.session_state.setdefault("_history_loaded_user", "")
st.session_state.setdefault("_prev_user_id", "")
history_store = get_history_store()

with st.sidebar:
    st.caption("本会话的对话仅保存在当前浏览器标签页，与其他访问者隔离。")
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
                st.session_state["chat_messages"] = [
                    {"role": m["role"], "content": m["content"]} for m in loaded
                ]
                st.session_state["_history_loaded_user"] = user_id
            else:
                st.session_state["chat_messages"] = []
            st.rerun()
    with col2:
        if st.button("清空本轮对话"):
            st.session_state["chat_messages"] = []
            st.rerun()

# 长期记忆：按 user_id 加载完整历史（切换用户且消息为空时）
user_id = (st.session_state.get("user_id") or "").strip()
if user_id and st.session_state["chat_messages"] == [] and st.session_state["_history_loaded_user"] != user_id:
    loaded = history_store.get_messages(user_id)
    if loaded:
        st.session_state["chat_messages"] = [
            {"role": m["role"], "content": m["content"]} for m in loaded
        ]
    st.session_state["_history_loaded_user"] = user_id

for message in st.session_state["chat_messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    # 长期记忆：持久化完整历史（仅当有 user_id）
    if user_id:
        history_store.append_message(user_id, "user", prompt)

    set_session_id(user_id)

    full_parts: list[str] = []
    with st.spinner("agent思考中..."):
        messages_for_agent = list(st.session_state["chat_messages"])

        def stream_chars():
            for piece in st.session_state["agent"].execute_stream(
                    messages_for_agent, user_id=user_id or None
            ):
                full_parts.append(piece)
                for char in piece:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(stream_chars())

    agent_inst = st.session_state["agent"]
    assistant_text = (
        (getattr(agent_inst, "last_turn_display_assistant_text", "") or "").strip()
        or "".join(full_parts)
    )
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": assistant_text}
    )
    # 长期记忆：持久化 assistant 回复（记忆抽取在 Agent 后端 execute_stream 结束并入队，见 memory/memory_queue.py）
    if user_id:
        history_store.append_message(user_id, "assistant", assistant_text)
    st.rerun()
