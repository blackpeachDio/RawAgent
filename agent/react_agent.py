"""
React Agent：支持多轮 messages；流式输出。
会话隔离：由调用方（如 app.py 的 st.session_state["chat_messages"]）按浏览器会话分别维护列表。
模型用长期记忆（摘要、画像）：由 Agent 内部按 user_id 检索并注入 context。
"""
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agent.react_graph_build import compile_react_agent
from rag.warmup import maybe_preload_rerank_cross_encoder
from utils.config_utils import agent_conf
from utils.agent_stream_display import (
    assistant_final_display_text,
    assistant_stream_visible_text,
)
from utils.log_utils import log_timing, logger
from utils.memory_inject import memory_inject_flags
from utils.memory_utils import trim_conversation_messages, validate_chat_messages
from memory.factual_multi import format_factual_block_for_injection
from memory.memory_queue import enqueue_memory_job


def _inject_memory_context(user_id: str, query: str) -> dict:
    """按 user_id 与配置注入：事实性画像（FactualStore）+ 向量记忆（经验/摘要/事件）。"""
    inject_factual, inject_vector = memory_inject_flags(query, agent_conf)
    if agent_conf.get("memory_inject_mode", "always").strip().lower() == "auto":
        logger.debug(
            "[memory] inject factual=%s vector=%s | q=%s",
            inject_factual,
            inject_vector,
            query[:80],
        )

    parts: list[str] = []
    if inject_factual:
        try:
            from memory.factual_store import get_factual_store

            factual = get_factual_store().get_all(user_id)
            if factual:
                block = format_factual_block_for_injection(
                    factual,
                    (query or "").strip(),
                    agent_conf,
                )
                if block.strip():
                    parts.append("【用户画像】\n" + block)
        except Exception:
            pass
    if inject_vector:
        try:
            from memory.chroma_memory import get_memory_store

            k_vec = int(agent_conf.get("memory_inject_vector_k", 5))
            vector_parts = get_memory_store().get_relevant(user_id, query, k=k_vec)
            if vector_parts and len(vector_parts) > 0:
                parts.append("【经验与摘要】\n" + "\n".join(vector_parts))
        except Exception:
            pass
    if parts:
        return {"memory": "\n\n".join(parts)}
    return {}


class ReactAgent:
    def __init__(self):
        self.agent = compile_react_agent()
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

    def _iter_assistant_stream(self, trimmed: list[dict], context: dict):
        input_dict = {"messages": trimmed}
        run_config = {"recursion_limit": self._recursion_limit}
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
                if len(text) > len(prev_assistant_text) and text.startswith(
                        prev_assistant_text
                ):
                    delta = text[len(prev_assistant_text):]
                    prev_assistant_text = text
                    if delta:
                        yield delta
                elif text != prev_assistant_text:
                    prev_assistant_text = text
                    yield text + ("\n" if not text.endswith("\n") else "")
        except GraphRecursionError as e:
            logger.warning(
                "[agent] 已达图执行步数上限 agent_recursion_limit=%s: %s",
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

    def execute_stream(self, messages: list[dict], user_id: str | None = None):
        """
        执行一轮对话（messages 须包含本轮及之前所有 user/assistant，最后一条须为当前 user）。

        Args:
            messages: 对话消息列表
            user_id: 当前用户标识；有则从长期记忆检索摘要/画像并注入 context

        Yields:
            str: 本轮助手流式增量（可含推理/思考过程，供前台实时展示）。
            会话持久化与 last_turn_display_assistant_text 为去掉推理后的正文。
            若开启 reflection：先发主回答流式增量，结束后再做审核；未达标时可选先生成「前置反思」，
            再将用户原问题 + 反思 + 审核要点一并注入，流式输出修正轮。
        """
        # 验证信息
        validate_chat_messages(messages)
        if not messages:
            raise ValueError("messages 不能为空")
        if messages[-1].get("role") != "user":
            raise ValueError("messages 最后一条须为 user")

        trimmed = trim_conversation_messages(messages, self._max_messages)
        # 本轮用户原文：记忆注入与自检都用它；修正轮时也不能改成审核反馈里的句子去检索
        original_query = (messages[-1].get("content") or "").strip()
        query_preview = original_query[:160] + ("…" if len(original_query) > 160 else "")
        log_timing("agent_turn", "start", query_preview=query_preview)
        all_emitted: list[str] = []
        self.last_turn_display_assistant_text = ""
        main_final_text = ""
        fix_final_text = ""
        user_notice_text = ""
        try:
            # 长期记忆注入（进 LangGraph context，由中间件拼进 system）
            ctx0 = self._build_context(user_id, original_query)

            # 主回答：LangGraph 流式增量（可含推理过程）；最终正文见 _last_turn_final_assistant_text
            draft_parts: list[str] = []
            for delta in self._iter_assistant_stream(trimmed, ctx0):
                draft_parts.append(delta)
                all_emitted.append(delta)
                yield delta
            main_final_text = (
                (getattr(self, "_last_turn_final_assistant_text", None) or "").strip()
                or "".join(draft_parts).strip()
            )
            draft = main_final_text
            log_timing("agent_turn", "main_stream_done", char_count=len(draft))

            # reflection 未开启，或主回答为空：不再自检、不加轮
            if not bool(agent_conf.get("reflection_enabled", False)):
                self.last_turn_display_assistant_text = main_final_text
                return
            if not draft.strip():
                self.last_turn_display_assistant_text = main_final_text
                return

            # 主回答流结束后的 LLM 自检（见 config/agent.yml reflection_*）
            from agent.reflect_critique import reflect_critique_score
            from agent.reflect_regenerate import build_regeneration_user_message, run_reflection_step

            min_score = float(agent_conf.get("reflection_min_score", 0.65))
            extra_turns = int(agent_conf.get("reflection_max_extra_turns", 1))
            score, reason = reflect_critique_score(original_query, draft)
            logger.info(
                "[reflection] score=%.3f min=%.3f reason=%s",
                score,
                min_score,
                (reason[:160] + "…") if len(reason) > 160 else reason,
            )
            # 分数达标或不允许额外轮次：结束
            if score >= min_score or extra_turns <= 0:
                self.last_turn_display_assistant_text = main_final_text
                return

            reflection_text = run_reflection_step(original_query, draft, reason)
            correction = build_regeneration_user_message(
                original_query,
                reflection_text,
                reason,
            )
            messages_retry = list(messages) + [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": correction},
            ]
            trimmed2 = trim_conversation_messages(messages_retry, self._max_messages)
            ctx1 = self._build_context(user_id, original_query)

            user_notice_text = (
                "\n\n---\n"
                "【提示】刚才的回答未通过质量自检，可能存在不准确、不完整或与问题不够贴切之处。"
                "正在重新生成回复，请稍候。\n\n"
            )
            all_emitted.append(user_notice_text)
            yield user_notice_text

            fix_parts: list[str] = [user_notice_text]
            for delta in self._iter_assistant_stream(trimmed2, ctx1):
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
                logger.warning("[agent] 记忆抽取入队失败: %s", e)
            log_timing("agent_turn", "end")


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
