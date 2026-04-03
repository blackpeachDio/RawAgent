import time

import streamlit as st

from agent.react_agent import ReactAgent
from memory.history_store import get_history_store
from utils.config_utils import agent_conf
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
    messages_for_agent = list(st.session_state["chat_messages"])
    show_tool_trace = bool(agent_conf.get("streamlit_show_tool_trace", True))
    _status_fn = getattr(st, "status", None)

    with st.chat_message("assistant"):
        trace_area = st.container()
        answer_ph = st.empty()
        if show_tool_trace and _status_fn is not None:
            with trace_area:
                with _status_fn("工具与推理步骤", expanded=True) as status:
                    for ev in st.session_state["agent"].iter_stream_events(
                            messages_for_agent, user_id=user_id or None
                    ):
                        if ev["type"] == "tool_call":
                            nm = ev.get("name") or "?"
                            ap = (ev.get("args_preview") or "").strip() or "（无参数）"
                            status.write(f"调用工具 **{nm}**  ·  参数: `{ap}`")
                        elif ev["type"] == "tool_result":
                            nm = ev.get("name") or "?"
                            cp = (ev.get("content_preview") or "").strip() or "（空）"
                            status.write(f"工具 **{nm}** 返回: `{cp}`")
                        elif ev["type"] == "text_delta":
                            full_parts.append(ev["content"])
                            answer_ph.markdown("".join(full_parts))
                        elif ev["type"] == "error":
                            full_parts.append(ev["content"])
                            answer_ph.markdown("".join(full_parts))
        elif show_tool_trace:
            with trace_area:
                with st.expander("工具与推理步骤", expanded=True):
                    for ev in st.session_state["agent"].iter_stream_events(
                            messages_for_agent, user_id=user_id or None
                    ):
                        if ev["type"] == "tool_call":
                            nm = ev.get("name") or "?"
                            ap = (ev.get("args_preview") or "").strip() or "（无参数）"
                            st.write(f"调用工具 **{nm}**  ·  `{ap}`")
                        elif ev["type"] == "tool_result":
                            nm = ev.get("name") or "?"
                            cp = (ev.get("content_preview") or "").strip() or "（空）"
                            st.write(f"**{nm}** 返回: `{cp}`")
                        elif ev["type"] == "text_delta":
                            full_parts.append(ev["content"])
                            answer_ph.markdown("".join(full_parts))
                        elif ev["type"] == "error":
                            full_parts.append(ev["content"])
                            answer_ph.markdown("".join(full_parts))
        else:
            with trace_area:
                with st.spinner("agent思考中..."):

                    def stream_chars():
                        for piece in st.session_state["agent"].execute_stream(
                                messages_for_agent, user_id=user_id or None
                        ):
                            full_parts.append(piece)
                            for char in piece:
                                time.sleep(0.01)
                                yield char

                    answer_ph.write_stream(stream_chars())

    assistant_text = "".join(full_parts)
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": assistant_text}
    )
    # 长期记忆：持久化 assistant 回复
    if user_id:
        history_store.append_message(user_id, "assistant", assistant_text)
        # 异步：提取事实与事件并分别存储到 FactualStore / ChromaMemoryStore
        try:
            from memory.extract_store import extract_and_store_async
            extract_and_store_async(user_id, prompt, assistant_text)
        except Exception:
            pass
    st.rerun()
