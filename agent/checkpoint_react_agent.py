"""
基于 LangGraph checkpoint 的 ReAct Agent（与 ReactAgent 并行，默认不被 app 引用）。

- 使用 compile_react_agent(checkpointer=...)，通过 configurable.thread_id 区分会话线程。
- 每轮调用只向图输入「本条用户消息」（HumanMessage），历史由 checkpoint 中的 messages 合并，避免与全量 messages 重复叠加。
- 长期记忆注入、reflection、记忆入队与 ReactAgent 对齐。

用法示例（伪代码）::

    from agent.checkpoint_react_agent import CheckpointReactAgent, make_checkpoint_thread_id

    agent = CheckpointReactAgent()
    tid = make_checkpoint_thread_id("alice", session_tag="tab-1")
    for chunk in agent.execute_stream(
        [{"role": "user", "content": "你好"}],
        user_id="alice",
        thread_id=tid,
    ):
        print(chunk, end="")
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from agent.react_graph_build import compile_react_agent
from agent.react_agent import _inject_memory_context
from rag.warmup import maybe_preload_rerank_cross_encoder
from memory.memory_queue import enqueue_memory_job
from utils.agent_stream_display import (
    assistant_final_display_text,
    assistant_stream_visible_text,
)
from utils.config_utils import agent_conf
from utils.log_utils import log_timing, logger
from utils.memory_utils import validate_chat_messages


def make_checkpoint_thread_id(user_id: str | None, session_tag: str | None = None) -> str:
    """
    生成 checkpoint 的 thread_id。同一 thread_id 共享图状态（含 messages）。

    - 仅 user_id：同一用户单会话场景可用。
    - 带 session_tag：多标签/多设备并行会话时避免互相覆盖（建议 Streamlit 侧传入稳定 session 标识）。
    """
    u = (user_id or "anonymous").strip() or "anonymous"
    st = (session_tag or "").strip()
    return f"{u}::{st}" if st else u


class CheckpointReactAgent:
    """compile_react_agent + checkpointer；stream 时传入 thread_id。"""

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer or MemorySaver()
        self.agent = compile_react_agent(checkpointer=self.checkpointer)
        self._max_messages = int(agent_conf.get("conversation_max_messages", 40))
        self._recursion_limit = int(agent_conf.get("agent_recursion_limit", 40))

        maybe_preload_rerank_cross_encoder()

    @staticmethod
    def _build_context(user_id: str | None, memory_query_for_inject: str) -> dict:
        q = (memory_query_for_inject or "").strip()
        context: dict = {"report": False, "original_query": q}
        if user_id:
            context.update(_inject_memory_context(user_id, q))
        return context

    def _iter_assistant_stream_checkpoint(
        self,
        user_message: HumanMessage,
        context: dict,
        thread_id: str,
    ):
        input_dict = {"messages": [user_message]}
        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }
        prev_assistant_text = ""
        last_ai: AIMessage | None = None
        try:
            stream_iter = self.agent.stream(
                input_dict,
                stream_mode="values",
                context=context,
                config=run_config,
            )
            for chunk in stream_iter:
                raw_msgs = chunk.get("messages") or []
                if not raw_msgs:
                    continue
                latest = raw_msgs[-1]
                if not isinstance(latest, AIMessage):
                    continue
                last_ai = latest
                text = assistant_stream_visible_text(latest).strip()
                if not text:
                    continue
                if len(text) > len(prev_assistant_text) and text.startswith(prev_assistant_text):
                    delta = text[len(prev_assistant_text) :]
                    prev_assistant_text = text
                    if delta:
                        yield delta
                elif text != prev_assistant_text:
                    prev_assistant_text = text
                    yield text + ("\n" if not text.endswith("\n") else "")
        except GraphRecursionError as e:
            logger.warning(
                "[agent][checkpoint] 已达图执行步数上限 agent_recursion_limit=%s: %s",
                self._recursion_limit,
                e,
            )
            yield (
                "\n\n[提示] 本轮智能体推理步数已达上限，已停止。"
                "若问题较复杂，可拆成更短的问题分步提问，或在配置中适当调大 agent_recursion_limit。"
            )
        finally:
            self._last_turn_final_assistant_text = (
                assistant_final_display_text(last_ai) if last_ai else ""
            )

    def execute_stream(
        self,
        messages: list[dict],
        user_id: str | None = None,
        *,
        thread_id: str | None = None,
        session_tag: str | None = None,
    ):
        """
        messages 仍兼容「全量列表」校验（最后一条须为 user），但**实际只把最后一条用户消息**送入带 checkpoint 的图。

        thread_id 优先；否则用 make_checkpoint_thread_id(user_id, session_tag)。
        """
        validate_chat_messages(messages)
        if not messages:
            raise ValueError("messages 不能为空")
        if messages[-1].get("role") != "user":
            raise ValueError("messages 最后一条须为 user")

        original_query = (messages[-1].get("content") or "").strip()
        tid = (thread_id or "").strip() or make_checkpoint_thread_id(user_id, session_tag)

        query_preview = original_query[:160] + ("…" if len(original_query) > 160 else "")
        log_timing("agent_turn", "start", query_preview=query_preview, checkpoint=True)
        all_emitted: list[str] = []
        self.last_turn_display_assistant_text = ""
        main_final_text = ""
        fix_final_text = ""
        user_notice_text = ""
        try:
            ctx0 = self._build_context(user_id, original_query)
            user_human = HumanMessage(content=original_query)

            draft_parts: list[str] = []
            for delta in self._iter_assistant_stream_checkpoint(user_human, ctx0, tid):
                draft_parts.append(delta)
                all_emitted.append(delta)
                yield delta
            main_final_text = (
                (getattr(self, "_last_turn_final_assistant_text", None) or "").strip()
                or "".join(draft_parts).strip()
            )
            draft = main_final_text
            log_timing("agent_turn", "main_stream_done", char_count=len(draft), checkpoint=True)

            if not bool(agent_conf.get("reflection_enabled", False)):
                self.last_turn_display_assistant_text = main_final_text
                return
            if not draft.strip():
                self.last_turn_display_assistant_text = main_final_text
                return

            from agent.reflect_critique import reflect_critique_score

            min_score = float(agent_conf.get("reflection_min_score", 0.65))
            extra_turns = int(agent_conf.get("reflection_max_extra_turns", 1))
            score, reason = reflect_critique_score(original_query, draft)
            logger.info(
                "[reflection][checkpoint] score=%.3f min=%.3f reason=%s",
                score,
                min_score,
                (reason[:160] + "…") if len(reason) > 160 else reason,
            )
            if score >= min_score or extra_turns <= 0:
                self.last_turn_display_assistant_text = main_final_text
                return

            correction = (
                f"审核反馈（仅供你改进回答，不要复述本句）：{reason or '质量不足'}。"
                "请输出修正后的完整回答，直接面向用户，不要提及审核或修改过程。"
            )
            ctx1 = self._build_context(user_id, original_query)

            user_notice_text = (
                "\n\n---\n"
                "【提示】刚才的回答未通过质量自检，可能存在不准确、不完整或与问题不够贴切之处。"
                "正在重新生成回复，请稍候。\n\n"
            )
            all_emitted.append(user_notice_text)
            yield user_notice_text

            fix_parts: list[str] = [user_notice_text]
            correction_msg = HumanMessage(content=correction)
            for delta in self._iter_assistant_stream_checkpoint(correction_msg, ctx1, tid):
                fix_parts.append(delta)
                all_emitted.append(delta)
                yield delta
            fix_final_text = (
                (getattr(self, "_last_turn_final_assistant_text", None) or "").strip()
                or "".join(fix_parts[1:]).strip()
            )
            self.last_turn_display_assistant_text = user_notice_text + fix_final_text
            log_timing(
                "agent_turn",
                "reflection_stream_done",
                char_count=len("".join(fix_parts)),
                checkpoint=True,
            )
        finally:
            if not self.last_turn_display_assistant_text.strip():
                self.last_turn_display_assistant_text = main_final_text or (
                    (getattr(self, "_last_turn_final_assistant_text", None) or "").strip()
                )
            try:
                persist = self.last_turn_display_assistant_text.strip()
                if user_id and persist:
                    enqueue_memory_job(user_id, original_query, persist)
            except Exception as e:
                logger.warning("[agent][checkpoint] 记忆抽取入队失败: %s", e)
            log_timing("agent_turn", "end", checkpoint=True)
