import time
import uuid

import streamlit as st
from agent.react_agent import ReactAgent
from utils.log_utils import set_session_id

# 标题
st.title("扫地机器人智能客服")
st.divider()

# 每个浏览器会话独立的 session_state → 对话隔离（不同用户/标签页互不共享 chat_messages）
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 为当前会话生成唯一标识，用户未输入时用于日志区分
if "_log_session_id" not in st.session_state:
    st.session_state["_log_session_id"] = str(uuid.uuid4())[:8]

# 兼容旧版仅存的 message 列表
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = st.session_state.pop("message", None) or []

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
    if "_prev_user_id" not in st.session_state:
        st.session_state["_prev_user_id"] = ""
    if user_id != st.session_state["_prev_user_id"]:
        st.session_state["_prev_user_id"] = user_id
        # 切换用户时清空当前对话，后续持久化可按 user_id 分别加载
        st.session_state["chat_messages"] = []
        st.rerun()

    if st.button("清空本轮对话"):
        st.session_state["chat_messages"] = []
        st.rerun()

for message in st.session_state["chat_messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    # 设置日志 session 标识：优先使用用户输入，便于多用户场景过滤及后续按 user_id 持久化
    session_id = (
        (st.session_state.get("user_id") or "").strip() or st.session_state["_log_session_id"]
    )
    set_session_id(session_id)

    full_parts: list[str] = []
    with st.spinner("智能客服思考中..."):
        messages_for_agent = list(st.session_state["chat_messages"])

        def stream_chars():
            for piece in st.session_state["agent"].execute_stream(messages_for_agent):
                full_parts.append(piece)
                for char in piece:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(stream_chars())

    assistant_text = "".join(full_parts)
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": assistant_text}
    )
    st.rerun()
