"""
React Agent：支持多轮 messages；流式输出。
会话隔离：由调用方（如 app.py 的 st.session_state["chat_messages"]）按浏览器会话分别维护列表。
模型用长期记忆（摘要、画像）：由 Agent 内部按 user_id 检索并注入 context。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain.agents import create_agent

from agent.mcp_loader import load_remote_mcp_tools_sync
from agent.values_stream_events import iter_stream_events_from_values_stream
from raw_agent_skillkit import build_skill_tools
from agent.tools.agent_tools import *
from agent.tools.middleware import *
from model.factory import chat_model
from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.memory_inject import memory_inject_flags
from utils.memory_utils import trim_conversation_messages, validate_chat_messages


def _inject_memory_context(user_id: str, query: str) -> dict:
    """按 user_id 与配置注入：事实性画像（FactualStore）+ 向量记忆（经验/摘要/事件）。"""
    inject_factual, inject_vector = memory_inject_flags(query, agent_conf)
    if agent_conf.get("memory_inject_mode", "always").strip().lower() == "auto":
        logger.debug(
            "[memory] inject factual=%s vector=%s | q=%s",
            inject_factual,
            inject_vector,
            query[:80],
        )

    parts: list[str] = []
    if inject_factual:
        try:
            from memory.factual_store import get_factual_store

            factual = get_factual_store().get_all(user_id)
            if factual:
                lines = [f"{k}: {v}" for k, v in sorted(factual.items())]
                parts.append("【用户画像】\n" + "\n".join(lines))
        except Exception:
            pass
    if inject_vector:
        try:
            from memory.chroma_memory import get_memory_store

            k_vec = int(agent_conf.get("memory_inject_vector_k", 5))
            vector_parts = get_memory_store().get_relevant(user_id, query, k=k_vec)
            if vector_parts and len(vector_parts) > 0:
                parts.append("【经验与摘要】\n" + "\n".join(vector_parts))
        except Exception:
            pass
    if parts:
        return {"memory": "\n\n".join(parts)}
    return {}


class ReactAgent:
    def __init__(self):
        mcp_tools = load_remote_mcp_tools_sync()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_id,
                get_user_location,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
                *build_skill_tools(),
                *mcp_tools,
            ],
            middleware=[monitor_tool, log_before_model, build_system_prompt, log_wrap_model_tokens],
        )
        self._max_messages = int(agent_conf.get("conversation_max_messages", 40))
        self._recursion_limit = int(agent_conf.get("agent_recursion_limit", 40))

    def _prepare_run(
            self,
            messages: list[dict],
            user_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        validate_chat_messages(messages)
        if not messages:
            raise ValueError("messages 不能为空")
        if messages[-1].get("role") != "user":
            raise ValueError("messages 最后一条须为 user")

        trimmed = trim_conversation_messages(messages, self._max_messages)
        input_dict = {"messages": trimmed}

        context: dict[str, Any] = {"report": False}
        if user_id:
            query = (messages[-1].get("content") or "").strip()
            context.update(_inject_memory_context(user_id, query))

        run_config = {"recursion_limit": self._recursion_limit}
        return input_dict, context, run_config

    def iter_stream_events(
            self,
            messages: list[dict],
            user_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        流式执行一轮对话，产出结构化事件，供 Streamlit 等展示工具链。

        Yields:
            {"type": "tool_call", "name": str, "args_preview": str, "id": str}
            {"type": "tool_result", "name": str, "content_preview": str, "tool_call_id": str}
            {"type": "text_delta", "content": str}
            {"type": "error", "content": str}
        """
        input_dict, context, run_config = self._prepare_run(messages, user_id)
        stream_iter = self.agent.stream(
            input_dict,
            stream_mode="values",
            context=context,
            config=run_config,
        )
        yield from iter_stream_events_from_values_stream(
            stream_iter,
            recursion_limit=self._recursion_limit,
        )

    def execute_stream(self, messages: list[dict], user_id: str | None = None):
        """
        执行一轮对话（messages 须包含本轮及之前所有 user/assistant，最后一条须为当前 user）。

        Yields:
            str: 本轮助手回复的增量文本片段（拼接后与最终回复一致）。
        """
        for ev in self.iter_stream_events(messages, user_id=user_id):
            if ev["type"] == "text_delta":
                yield ev["content"]
            elif ev["type"] == "error":
                yield ev["content"]


if __name__ == "__main__":
    agent = ReactAgent()
    history: list[dict] = [{"role": "user", "content": "水箱加水后漏水怎么处理"}]
    parts: list[str] = []
    for piece in agent.execute_stream(history):
        print(piece, end="", flush=True)
        parts.append(piece)
    print()