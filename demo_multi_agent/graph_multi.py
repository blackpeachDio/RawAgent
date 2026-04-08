"""
多 Agent 示例：supervisor 路由 → math（ReAct+工具）或 general（纯对话）。

流程：START → supervisor（只写 route 字段）→ math_agent | general_agent → END
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from demo_agent.graph_agent import build_react_graph, default_tools
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str


def _supervisor_node(model):
    """根据最后一条用户消息输出 MATH / GENERAL，写入 state['route']。"""

    def node(state: MultiAgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, HumanMessage):
            return {"route": "general"}
        r = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是路由模块。只输出一个词，不要标点或其它解释。\n"
                        "输出 MATH：用户需要数字、长度、字符数、计算、工具测量等。\n"
                        "输出 GENERAL：闲聊、翻译、常识、无需计算与工具。"
                    ),
                ),
                last,
            ]
        )
        text = (r.content or "").strip().upper()
        first = text.split()[0] if text else ""
        route: Literal["math", "general"] = "math" if first == "MATH" else "general"
        return {"route": route}

    return node


def _math_agent_node(model):
    """复用单 Agent ReAct 子图，只把子图产生的新消息追加到主 state。"""

    sub = build_react_graph(
        model,
        tools=default_tools(),
        system_prompt="你是数学与计算助手，需要时用 get_word_length 等工具，回答简洁。",
    )

    def node(state: MultiAgentState) -> dict:
        out = sub.invoke(
            {"messages": state["messages"]},
            config={"recursion_limit": 25},
        )
        n = len(state["messages"])
        tail = out["messages"][n:]
        return {"messages": tail}

    return node


def _general_agent_node(model):
    def node(state: MultiAgentState) -> dict:
        msgs = list(state["messages"])
        r = model.invoke(
            [
                SystemMessage(content="你是通用助手，直接回答用户，不要用工具。"),
                *msgs,
            ]
        )
        return {"messages": [r]}

    return node


def build_multi_agent_graph(model):
    """
    supervisor → math_agent（ReAct+工具）或 general_agent（无工具）。
    invoke 示例：{"messages": [HumanMessage("...")], "route": ""}
    """
    builder = StateGraph(MultiAgentState)
    builder.add_node("supervisor", _supervisor_node(model))
    builder.add_node("math_agent", _math_agent_node(model))
    builder.add_node("general_agent", _general_agent_node(model))
    builder.add_edge(START, "supervisor")

    def _route(state: MultiAgentState) -> str:
        r = state.get("route") or "general"
        return r if r in ("math", "general") else "general"

    builder.add_conditional_edges(
        "supervisor",
        _route,
        {"math": "math_agent", "general": "general_agent"},
    )
    builder.add_edge("math_agent", END)
    builder.add_edge("general_agent", END)
    return builder.compile()
