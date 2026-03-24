"""
记忆提取与存储：一轮对话结束后，异步调用 LLM 提取事实与事件并分别存储。
"""
from __future__ import annotations

import json
import threading

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from utils.log_utils import logger
from utils.prompt_utils import load_mem_extract_prompts

ALLOWED_FACT_KEYS = frozenset(
    {"name", "age", "job", "city", "hobby", "character", "preferences", "avoid", "rules"}
)


def _format_conversation(user_msg: str, assistant_msg: str) -> str:
    """将本轮对话格式化为文本。"""
    return f"用户：{user_msg}\n助手：{assistant_msg}"


def _extract_facts_and_events(conversation: str) -> tuple[list[dict], list[str]]:
    """
    调用 LLM 提取事实与事件，返回 (facts, events)。

    facts: [{"key": "hobby", "value": "香蕉"}, ...]
    events: ["用户询问了水箱漏水问题", ...]
    """
    prompt_text = load_mem_extract_prompts()
    template = PromptTemplate.from_template(prompt_text)
    chain = template | chat_model | StrOutputParser()
    try:
        raw = chain.invoke({"conversation": conversation})
    except Exception as e:
        logger.warning("[Memory] LLM 提取失败: %s", e)
        return [], []

    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
        facts = data.get("facts") or []
        events = data.get("events") or []
        if not isinstance(facts, list):
            facts = []
        if not isinstance(events, list):
            events = [str(events)] if events else []
        # 过滤非法 fact key
        facts = [
            {"key": f["key"], "value": str(f.get("value", ""))}
            for f in facts
            if isinstance(f, dict) and f.get("key") in ALLOWED_FACT_KEYS
        ]
        events = [str(e).strip() for e in events if e]
        return facts, events
    except json.JSONDecodeError as e:
        logger.warning("[Memory] 解析提取结果失败: %s | raw=%s", e, raw[:200])
        return [], []


def _store_extracted(user_id: str, facts: list[dict], events: list[str]) -> None:
    """将提取结果写入 FactualStore 与 ChromaMemoryStore。"""
    if not user_id:
        return
    try:
        from memory.factual_store import get_factual_store
        from memory.chroma_memory import get_memory_store
    except ImportError as e:
        logger.warning("[Memory] 导入存储失败: %s", e)
        return

    factual = get_factual_store()
    vector_store = get_memory_store()

    for f in facts:
        k, v = f.get("key", ""), f.get("value", "")
        if k and v:
            factual.set(user_id, k, v)
            logger.debug("[Memory] 写入事实 user_id=%s key=%s value=%s", user_id, k, v)

    for evt in events:
        if evt:
            vector_store.add(user_id, evt, memory_type="event")
            logger.debug("[Memory] 写入事件 user_id=%s event=%s", user_id, evt[:50])


def extract_and_store(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    同步：提取并存储。供异步线程调用。

    Args:
        user_id: 用户标识
        user_msg: 本轮用户输入
        assistant_msg: 本轮助手回复
    """
    if not user_id:
        return
    conversation = _format_conversation(user_msg, assistant_msg)
    facts, events = _extract_facts_and_events(conversation)
    if facts or events:
        _store_extracted(user_id, facts, events)
        logger.info(
            "[Memory] 提取完成 user_id=%s facts=%s events=%d",
            user_id,
            json.dumps(facts, ensure_ascii=False),
            len(events),
        )
    else:
        logger.debug("[Memory] 本轮无可用提取 user_id=%s", user_id)


def extract_and_store_async(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """异步：在后台线程执行提取与存储，不阻塞主流程。"""
    if not user_id:
        return
    t = threading.Thread(
        target=extract_and_store,
        args=(user_id, user_msg, assistant_msg),
        name="memory-extract",
        daemon=True,
    )
    t.start()
    logger.debug("[Memory] 已启动异步提取线程 user_id=%s", user_id)
