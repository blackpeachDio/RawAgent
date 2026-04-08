"""
在项目根目录执行：python -m demo_agent.demo
需已配置 config 与 DashScope（与主工程一致）。
"""
from langchain_core.messages import HumanMessage

from demo_agent.graph_agent import build_react_graph, default_tools
from model.factory import chat_model


def main():
    graph = build_react_graph(
        chat_model,
        tools=default_tools(),
        system_prompt="你是助手；需要时用工具回答。",
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="'hello' 有几个字符？用工具算。")]},
        config={"recursion_limit": 25},
    )
    for m in result["messages"]:
        extra = getattr(m, "tool_calls", None) or ""
        print(m.type, (m.content or "")[:200], extra)


if __name__ == "__main__":
    main()
