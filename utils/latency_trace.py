"""
单次用户提问的耗时追踪：用 ContextVar 在同一线程内串联 Agent / 工具 / RAG。

日志前缀统一为 [latency]，便于 grep 与后续把日志交给分析。
"""
from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from utils.log_utils import logger

_turn: contextvars.ContextVar[TurnLatency | None] = contextvars.ContextVar(
    "turn_latency", default=None
)


@dataclass
class TurnLatency:
    trace_id: str
    t0: float
    wall_iso: str
    query_preview: str
    model_before_count: int = 0
    # 累计：主流程 + 可能的 reflection 多轮
    llm_api_total_s: float = 0.0
    llm_api_calls: int = 0
    rag_retrieve_calls: int = 0
    tags: dict[str, Any] = field(default_factory=dict)


def active_turn() -> TurnLatency | None:
    return _turn.get()


def trace_id_or_dash() -> str:
    t = _turn.get()
    return t.trace_id if t else "-"


def start_turn(query_preview: str, max_q: int = 160) -> str:
    """开始一轮用户提问追踪；返回 trace_id。"""
    tid = uuid.uuid4().hex[:12]
    q = (query_preview or "").strip().replace("\n", " ")
    if len(q) > max_q:
        q = q[:max_q] + "…"
    tr = TurnLatency(
        trace_id=tid,
        t0=time.perf_counter(),
        wall_iso=datetime.now().isoformat(timespec="milliseconds"),
        query_preview=q,
    )
    _turn.set(tr)
    logger.info(
        "[latency] trace=%s phase=turn_start wall=%s dt_s=0.0000 | q=%s",
        tid,
        tr.wall_iso,
        q,
    )
    return tid


def dt_from_turn_start() -> float:
    t = _turn.get()
    if not t:
        return 0.0
    return time.perf_counter() - t.t0


def note_before_model() -> None:
    """LangGraph before_model：每次即将调用 LLM 前触发。"""
    tr = _turn.get()
    if not tr:
        return
    tr.model_before_count += 1
    dt = time.perf_counter() - tr.t0
    phase = "first_llm_scheduled" if tr.model_before_count == 1 else "llm_scheduled"
    logger.info(
        "[latency] trace=%s phase=%s idx=%d dt_from_turn_start_s=%.4f wall=%s",
        tr.trace_id,
        phase,
        tr.model_before_count,
        dt,
        datetime.now().isoformat(timespec="milliseconds"),
    )


def note_llm_api_wall(wall_s: float) -> None:
    """wrap_model_call 内一次真实模型调用耗时。"""
    tr = _turn.get()
    if not tr:
        logger.info(
            "[latency] trace=- phase=llm_api wall_s=%.4f (no active turn trace)",
            wall_s,
        )
        return
    tr.llm_api_calls += 1
    tr.llm_api_total_s += wall_s
    logger.info(
        "[latency] trace=%s phase=llm_api idx=%d wall_s=%.4f dt_from_turn_start_s=%.4f",
        tr.trace_id,
        tr.llm_api_calls,
        wall_s,
        time.perf_counter() - tr.t0,
    )


def note_assistant_stream_done(phase: str, char_len: int) -> None:
    """主回答或修正轮：流式增量已全部产出（尚未算用户侧 UI）。"""
    tr = _turn.get()
    if not tr:
        return
    logger.info(
        "[latency] trace=%s phase=assistant_stream_done sub=%s chars=%d dt_from_turn_start_s=%.4f wall=%s",
        tr.trace_id,
        phase,
        char_len,
        time.perf_counter() - tr.t0,
        datetime.now().isoformat(timespec="milliseconds"),
    )


def note_tool_done(tool_name: str, wall_s: float) -> None:
    tr = _turn.get()
    tid = tr.trace_id if tr else "-"
    logger.info(
        "[latency] trace=%s phase=tool name=%s wall_s=%.4f dt_from_turn_start_s=%.4f",
        tid,
        tool_name,
        wall_s,
        dt_from_turn_start() if tr else 0.0,
    )


def note_rag_retrieve_breakdown(
        *,
        rewrite_on: bool,
        rewrite_s: float,
        recall_s: float,
        rerank_on: bool,
        rerank_s: float,
        expand_s: float,
        pool_len: int,
        out_len: int,
) -> None:
    tr = _turn.get()
    tid = tr.trace_id if tr else "-"
    if tr:
        tr.rag_retrieve_calls += 1
        idx = tr.rag_retrieve_calls
    else:
        idx = 0
    logger.info(
        "[latency] trace=%s phase=rag_retrieve idx=%d rewrite_on=%s rewrite_s=%.4f recall_pool_s=%.4f "
        "rerank_on=%s rerank_s=%.4f expand_parent_s=%.4f pool=%d out=%d dt_from_turn_start_s=%.4f",
        tid,
        idx,
        rewrite_on,
        rewrite_s,
        recall_s,
        rerank_on,
        rerank_s,
        expand_s,
        pool_len,
        out_len,
        dt_from_turn_start() if tr else 0.0,
    )


def note_rag_generate(wall_s: float) -> None:
    tr = _turn.get()
    tid = tr.trace_id if tr else "-"
    logger.info(
        "[latency] trace=%s phase=rag_summarize_llm wall_s=%.4f dt_from_turn_start_s=%.4f",
        tid,
        wall_s,
        dt_from_turn_start() if tr else 0.0,
    )


def end_turn() -> None:
    tr = _turn.get()
    if not tr:
        return
    total = time.perf_counter() - tr.t0
    logger.info(
        "[latency] trace=%s phase=turn_end dt_total_s=%.4f llm_api_calls=%d llm_api_sum_s=%.4f wall=%s | q=%s",
        tr.trace_id,
        total,
        tr.llm_api_calls,
        tr.llm_api_total_s,
        datetime.now().isoformat(timespec="milliseconds"),
        tr.query_preview,
    )
    _turn.set(None)
