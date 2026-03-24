"""
React Agent：支持多轮 messages；流式输出。
会话隔离：由调用方（如 app.py 的 st.session_state["chat_messages"]）按浏览器会话分别维护列表。
模型用长期记忆（摘要、画像）：由 Agent 内部按 user_id 检索并注入 context。
"""
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.tools.agent_tools import *
from agent.tools.middleware import *
from model.factory import chat_model
from utils.config_utils import agent_conf
from utils.memory_utils import trim_conversation_messages, validate_chat_messages


def _inject_memory_context(user_id: str, query: str) -> dict:
    """按 user_id 与当前问题检索模型用记忆（摘要、画像），注入 context。"""
    try:
        from memory.chroma_memory import get_memory_store

        memory_parts = get_memory_store().get_relevant(user_id, query, k=5)
        if memory_parts:
            return {"memory": "\n".join(memory_parts)}
    except Exception:
        pass
    return {}


class ReactAgent:
    def __init__(self):
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
            ],
            middleware=[monitor_tool, log_before_model, build_system_prompt],
        )
        self._max_messages = int(agent_conf.get("conversation_max_messages", 40))

    def execute_stream(self, messages: list[dict], user_id: str | None = None):
        """
        执行一轮对话（messages 须包含本轮及之前所有 user/assistant，最后一条须为当前 user）。

        Args:
            messages: 对话消息列表
            user_id: 当前用户标识；有则从长期记忆检索摘要/画像并注入 context

        Yields:
            str: 本轮助手回复的增量文本片段（拼接后与最终回复一致）。
        """
        validate_chat_messages(messages)
        if not messages:
            raise ValueError("messages 不能为空")
        if messages[-1].get("role") != "user":
            raise ValueError("messages 最后一条须为 user")

        trimmed = trim_conversation_messages(messages, self._max_messages)
        input_dict = {"messages": trimmed}

        context = {"report": False}
        if user_id:
            query = (messages[-1].get("content") or "").strip()
            context.update(_inject_memory_context(user_id, query))

        prev_assistant_text = ""
        for chunk in self.agent.stream(
                input_dict, stream_mode="values", context=context
        ):
            raw_msgs = chunk.get("messages") or []
            if not raw_msgs:
                continue
            latest = raw_msgs[-1]
            if not isinstance(latest, AIMessage):
                continue
            text = (latest.content or "").strip()
            if not text:
                continue
            # 增量：适配「内容逐步变长」的流式 AIMessage
            if len(text) > len(prev_assistant_text) and text.startswith(
                    prev_assistant_text
            ):
                delta = text[len(prev_assistant_text):]
                prev_assistant_text = text
                if delta:
                    yield delta
            elif text != prev_assistant_text:
                # 非前缀增长时退化为整段输出一次
                prev_assistant_text = text
                yield text + ("\n" if not text.endswith("\n") else "")


if __name__ == "__main__":
    agent = ReactAgent()
    history: list[dict] = [{"role": "user", "content": "水箱加水后漏水怎么处理"}]
    parts: list[str] = []
    for piece in agent.execute_stream(history):
        print(piece, end="", flush=True)
        parts.append(piece)
    print()
    history.append({"role": "assistant", "content": "".join(parts)})
    history.append({"role": "user", "content": "刚才说的第一步再讲细一点"})
    for piece in agent.execute_stream(history):
        print(piece, end="", flush=True)
