"""
编译 LangChain create_agent 图（与是否使用 checkpoint 无关）。
供 ReactAgent、CheckpointReactAgent 共用，避免两套工具/中间件分叉。
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from typing import TYPE_CHECKING

from agent.mcp_loader import load_remote_mcp_tools_sync
from tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    after_model,
    build_system_prompt,
    log_before_model,
    log_wrap_model_tokens,
    monitor_tool,
)
from model.factory import chat_model, turbo_model
from raw_agent_skillkit import build_skill_tools
from utils.prompt_utils import load_system_prompts

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def compile_react_agent(*, checkpointer=None) -> "CompiledStateGraph":
    """
    构建与线上一致的 ReAct 图；传入 checkpointer 时状态按 thread_id 持久化（见 CheckpointReactAgent）。
    """
    mcp_tools = load_remote_mcp_tools_sync()
    return create_agent(
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
        middleware=[
            monitor_tool,
            log_before_model,
            build_system_prompt,
            log_wrap_model_tokens,
            after_model,
            SummarizationMiddleware(
                model=turbo_model,
                trigger=("messages", 5),
                keep=("messages", 2),
                summary_prompt="请用简洁中文，客观总结下面的对话内容：\n\n{messages}",
            ),
        ],
        checkpointer=checkpointer,
    )
