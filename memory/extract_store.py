"""
记忆提取与存储：一轮对话结束后，异步调用 LLM 提取事实与事件并分别存储。
支持从最近 N 轮对话构建上下文，便于跨轮抽取事实与事件。
"""
from __future__ import annotations

import json
import threading
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from memory.factual_multi import MULTI_FACT_KEYS, merge_append_multi
from model.factory import chat_model
from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.prompt_utils import load_mem_extract_prompts

ALLOWED_FACT_KEYS = frozenset(
    {"name", "age", "job", "city", "hobby", "character", "preferences", "avoid", "rules"}
)


def _format_conversation(user_msg: str, assistant_msg: str) -> str:
    """单轮：将本轮对话格式化为文本。"""
    return f"【本轮】\n用户：{user_msg}\n助手：{assistant_msg}"


def _messages_to_user_assistant_rounds(msgs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """将 history 消息列表合并为 (user, assistant) 轮次列表（时间正序）。"""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(msgs):
        if msgs[i].get("role") != "user":
            i += 1
            continue
        u = str(msgs[i].get("content") or "").strip()
        if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
            a = str(msgs[i + 1].get("content") or "").strip()
            out.append((u, a))
            i += 2
        else:
            i += 1
    return out


def _format_rounds_for_prompt(rounds: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    base = len(rounds)
    for idx, (u, a) in enumerate(rounds):
        turn_no = idx + 1
        lines.append(f"【第{turn_no}轮】（共{base}轮，自旧到新）")
        lines.append(f"用户：{u}")
        lines.append(f"助手：{a}")
        lines.append("")
    return "\n".join(lines).strip()


def _truncate_rounds_by_char_budget(
        rounds: list[tuple[str, str]],
        max_chars: int,
) -> list[tuple[str, str]]:
    """从最早轮开始丢弃，直到总长度不超过 max_chars。"""
    r = list(rounds)
    while r and len(_format_rounds_for_prompt(r)) > max_chars:
        r.pop(0)
    return r


def _build_extraction_conversation(user_id: str, user_msg: str, assistant_msg: str) -> str:
    """
    构建写入 mem_extract 提示词的对话块。
    有 user_id 时优先从 FileHistoryStore 取最近 N 轮（已含本轮，因 assistant 已持久化后再异步提取）。
    """
    max_rounds = max(1, int(agent_conf.get("memory_extract_max_rounds", 5)))
    max_chars = max(500, int(agent_conf.get("memory_extract_max_context_chars", 12000)))

    if user_id:
        try:
            from memory.history_store import get_history_store

            msgs = get_history_store().get_messages(user_id)
            rounds = _messages_to_user_assistant_rounds(msgs)
            if rounds:
                tail = rounds[-max_rounds:]
                tail = _truncate_rounds_by_char_budget(tail, max_chars)
                return _format_rounds_for_prompt(tail)
        except Exception as e:
            logger.warning("[Memory] 读取多轮历史失败，退化为单轮: %s", e)

    return _format_conversation(user_msg, assistant_msg)


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
        # 过滤非法 fact key；hobby/character/preferences 可带 scenario（场景标签）
        facts = []
        for f in facts:
            if not isinstance(f, dict) or f.get("key") not in ALLOWED_FACT_KEYS:
                continue
            facts.append(
                {
                    "key": f["key"],
                    "value": str(f.get("value", "")),
                    "scenario": str(f.get("scenario") or "").strip(),
                }
            )
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

    max_multi = max(4, int(agent_conf.get("factual_multi_max_per_key", 24) or 24))
    snap = factual.get_all(user_id)

    for f in facts:
        k = f.get("key", "")
        v = str(f.get("value", "")).strip()
        scenario = str(f.get("scenario") or "").strip()
        if not k or not v:
            continue
        if k in MULTI_FACT_KEYS:
            prev = snap.get(k, "")
            merged = merge_append_multi(prev, v, scenario, max_per_key=max_multi)
            factual.set(user_id, k, merged)
            snap[k] = merged
            logger.debug(
                "[Memory] 合并多条事实 user_id=%s key=%s value=%s scenario=%s",
                user_id,
                k,
                v[:80],
                scenario or "-",
            )
        else:
            factual.set(user_id, k, v)
            snap[k] = v
            logger.debug("[Memory] 写入事实 user_id=%s key=%s value=%s", user_id, k, v[:120])

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
    conversation = _build_extraction_conversation(user_id, user_msg, assistant_msg)
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
