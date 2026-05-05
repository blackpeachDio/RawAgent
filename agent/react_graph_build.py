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
    request_user_clarification,
)
from tools.middleware import (
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

_HITL_SYSTEM_APPEND = """

### 人机追问（仅当图已启用 checkpoint 时可用工具 request_user_clarification）
当用户表述缺少必要信息、无法唯一确定工具入参或存在多种合理解读时，**应调用 `request_user_clarification`**，在 `questions` 中用**固定中文**写清需要用户补充的内容（可含简短编号列表）；勿在缺少信息时猜测或编造事实。
非缺信息场景不要调用该工具；其它工具能直接解决时不要滥用追问。
"""


def compile_react_agent(*, checkpointer=None) -> "CompiledStateGraph":
    """
    构建与线上一致的 ReAct 图；传入 checkpointer 时状态按 thread_id 持久化（见 CheckpointReactAgent）。
    """
    mcp_tools = load_remote_mcp_tools_sync()
    base_tools: list = [
        rag_summarize,
        get_weather,
        get_user_id,
        get_user_location,
        get_current_month,
        fetch_external_data,
        fill_context_for_report,
        *build_skill_tools(),
    ]
    if checkpointer is not None:
        base_tools.append(request_user_clarification)
    system_prompt = load_system_prompts()
    if checkpointer is not None:
        system_prompt = system_prompt + _HITL_SYSTEM_APPEND
    return create_agent(
        model=chat_model,
        system_prompt=system_prompt,
        tools=[
            *base_tools,
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
                trigger=("messages", 10),
                keep=("messages", 5),
                summary_prompt="请用简洁中文，客观总结下面的对话内容：\n\n{messages}",
            ),
        ],
        checkpointer=checkpointer,
    )
