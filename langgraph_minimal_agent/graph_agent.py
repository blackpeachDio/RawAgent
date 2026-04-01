"""
最小 LangGraph ReAct 风格 agent：
- MessagesState：对话消息列表
- agent 节点：绑定 tools 的模型调用
- tools 节点：ToolNode 执行 tool_calls
- 条件边：有 tool_calls → tools → 回到 agent；否则结束
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_word_length(text: str) -> int:
    """返回文本的字符长度。"""
    return len(text)


def default_tools():
    return [get_word_length]


def build_react_graph(model, tools: list | None = None, system_prompt: str | None = None):
    """
    构建编译后的 StateGraph。
    tools: LangChain BaseTool 列表；为 None 时使用示例工具 get_word_length。
    system_prompt: 若设置，在首条消息前注入一条 SystemMessage（多轮循环内只注入一次）。
    """
    tools = tools or default_tools()
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = list(state["messages"])
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    # 无 tool_calls → END；有则 → "tools"（END 由 tools_condition 内部指向，无需再 add_edge(..., END)）
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()
