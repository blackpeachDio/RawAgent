import time

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("扫地机器人智能客服")
st.divider()

# 每个浏览器会话独立的 session_state → 对话隔离（不同用户/标签页互不共享 chat_messages）
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 兼容旧版仅存的 message 列表
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = st.session_state.pop("message", None) or []

with st.sidebar:
    st.caption("本会话的对话仅保存在当前浏览器标签页，与其他访问者隔离。")
    if st.button("清空本轮对话"):
        st.session_state["chat_messages"] = []
        st.rerun()

for message in st.session_state["chat_messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

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
