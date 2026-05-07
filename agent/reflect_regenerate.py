"""审核低分后：先写「前置反思」，再带着用户原问题与反思做二次生成（由 ReactAgent / CheckpointReactAgent 调用）。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.reflect_critique import _get_reflection_llm, _normalize_llm_text
from utils.config_utils import agent_conf
from utils.log_utils import log_timing, logger
from utils.prompt_utils import load_reflect_step_prompts

_PLACEHOLDER_Q = "<<<REFLECT_QUERY>>>"
_PLACEHOLDER_DRAFT = "<<<DRAFT_ANSWER>>>"
_PLACEHOLDER_REASON = "<<<CRITIQUE_REASON>>>"


def run_reflection_step(
        reflect_query_text: str,
        draft_answer: str,
        critique_reason: str,
) -> str:
    """
    低分后、二次生成前：基于用户问题（可与审核用 query 一致，含 resume 时的补充）与审核要点，生成一段短反思。
    关闭 reflection_reasoning_enabled 或调用失败时返回空串。
    """
    if not bool(agent_conf.get("reflection_reasoning_enabled", True)):
        return ""
    template = load_reflect_step_prompts()
    body = (
        template.replace(_PLACEHOLDER_Q, (reflect_query_text or "").strip())
        .replace(_PLACEHOLDER_DRAFT, draft_answer or "")
        .replace(_PLACEHOLDER_REASON, (critique_reason or "").strip())
    )
    try:
        llm = _get_reflection_llm()
        log_timing("reflection_reasoning", "llm_start")
        out = llm.invoke([HumanMessage(content=body)])
        log_timing("reflection_reasoning", "llm_done")
        text = _normalize_llm_text(getattr(out, "content", out)).strip()
        if not text:
            logger.warning("[reflection] 前置反思为空，二次生成将仅带审核要点与原问题")
        return text
    except Exception as e:
        logger.warning("[reflection] 前置反思生成失败，二次生成将仅带审核要点与原问题: %s", e)
        return ""


def build_regeneration_user_message(
        query_anchor: str,
        reflection_text: str,
        critique_reason: str,
) -> str:
    """
    二次生成时注入给模型的 user 内容：显式包含用户原问题、可选反思、审核要点。
    query_anchor 在 resume 场景下应含原问+用户补充（与 reflect_critique 用的一致）。
    """
    q = (query_anchor or "").strip()
    reason_block = (critique_reason or "").strip() or "质量不足"
    refl = (reflection_text or "").strip()
    lines: list[str] = [
        "下面进行二次生成。请严格围绕「用户原问题」作答。",
        "",
        "【用户原问题】",
        q,
        "",
    ]
    if refl:
        lines.extend(
            [
                "【前置反思】（内化改进，勿逐字复述给用户）",
                refl,
                "",
            ]
        )
    lines.extend(
        [
            "【审核要点】",
            reason_block,
            "",
            "请输出修正后的完整回答，直接面向用户，不要提及审核、反思或修改过程。",
        ]
    )
    return "\n".join(lines)
