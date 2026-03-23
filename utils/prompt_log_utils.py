import os
from typing import Any, Callable


def get_prompt_log_config(default_max_chars: int = 20000) -> tuple[bool, int]:
    """
    控制 prompt 日志的输出量。

    环境变量：
    - RAWAGENT_PROMPT_LOG_FULL=1：不截断
    - RAWAGENT_PROMPT_LOG_MAX_CHARS=N：截断到 N 字符（默认 20000）
    """
    max_chars_env = os.environ.get("RAWAGENT_PROMPT_LOG_MAX_CHARS", "").strip()
    full = os.environ.get("RAWAGENT_PROMPT_LOG_FULL", "").strip() == "1"
    max_chars = int(max_chars_env) if max_chars_env.isdigit() else default_max_chars
    return full, max_chars


def maybe_truncate(text: str, *, full: bool, max_chars: int) -> str:
    if full or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def format_messages_as_prompt_text(
    messages: list[Any],
    *,
    truncate_fn: Callable[[str], str],
) -> str:
    """
    把 agent messages 结构化为便于排查的纯文本块。
    """
    parts: list[str] = []
    for i, m in enumerate(messages or []):
        m_type = type(m).__name__
        content = getattr(m, "content", None)
        if content is None:
            content_str = str(m)
        else:
            content_str = str(content).strip()
        parts.append(f"--- message[{i}] {m_type} ---\n{truncate_fn(content_str)}")
    return "\n\n".join(parts)


def log_truncated_block(logger: Any, begin_tag: str, end_tag: str, text: str) -> None:
    """
    统一打印“可截断的文本块”，减少业务代码重复拼接。
    """
    full, max_chars = get_prompt_log_config()
    truncated = maybe_truncate(text, full=full, max_chars=max_chars)
    logger.info("%s\n%s\n%s", begin_tag, truncated, end_tag)

