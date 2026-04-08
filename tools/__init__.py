"""
内建 Agent 工具的稳定入口：实现位于 ``agent.tools.agent_tools``，此处 re-export 便于统一 import。

业务侧可逐步改为 ``from tools import rag_summarize, ...`` 与 ``agent.tools`` 等价。
"""
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)

__all__ = [
    "fetch_external_data",
    "fill_context_for_report",
    "get_current_month",
    "get_user_id",
    "get_user_location",
    "get_weather",
    "rag_summarize",
]
