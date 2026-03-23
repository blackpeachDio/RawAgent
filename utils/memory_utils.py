"""对话记忆工具：截断、校验（与 Streamlit session 配合实现会话隔离）。"""


def trim_conversation_messages(messages: list[dict], max_count: int) -> list[dict]:
    """
    保留末尾 max_count 条消息（user/assistant 字典），避免上下文无限增长。
    max_count <= 0 表示不截断。
    """
    if max_count <= 0 or len(messages) <= max_count:
        return list(messages)
    return list(messages[-max_count:])


def validate_chat_messages(messages: list[dict]) -> None:
    """基本校验，避免把错误结构传入 Agent。"""
    if not isinstance(messages, list):
        raise TypeError("messages 必须为 list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise TypeError(f"messages[{i}] 必须为 dict")
        if "role" not in m or "content" not in m:
            raise ValueError(f"messages[{i}] 须含 role 与 content 字段")
