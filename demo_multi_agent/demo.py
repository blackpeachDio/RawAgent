"""
多 Agent 演示：在项目根目录执行
  python -m demo_multi_agent.demo
需已配置 DashScope（与主工程一致）。
"""
from langchain_core.messages import HumanMessage

from demo_multi_agent.graph_multi import build_multi_agent_graph
from model.factory import chat_model


def _print_messages(result: dict) -> None:
    for m in result["messages"]:
        extra = getattr(m, "tool_calls", None) or ""
        preview = (m.content or "")[:300]
        print(m.type, preview, extra)


def main():
    graph = build_multi_agent_graph(chat_model)

    print("=== 示例1：应路由到 math（需要长度/工具）===")
    r1 = graph.invoke(
        {
            "messages": [
                HumanMessage(content="'hello' 有几个字符？必须用工具算长度。"),
            ],
            "route": "",
        },
        config={"recursion_limit": 40},
    )
    _print_messages(r1)
    print("route=", r1.get("route"))

    print("\n=== 示例2：应路由到 general（闲聊）===")
    r2 = graph.invoke(
        {
            "messages": [HumanMessage(content="用一句话介绍什么是 LangGraph。")],
            "route": "",
        },
        config={"recursion_limit": 40},
    )
    _print_messages(r2)
    print("route=", r2.get("route"))


if __name__ == "__main__":
    main()
