"""
LLM 自建打分：用验证模型对比 ground_truth 与 agent 生成的答案，输出 0~1 分数。

与 RAG 精排打分不同：
- 精排（BGE）：query 与 document 的语义相关性，用于检索重排序
- 本模块：agent 答案与标准答案的吻合度，用于整体回答质量评估
"""
from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from utils.prompt_utils import load_judge_prompts


def llm_answer_score(
        judge_model: BaseChatModel,
        question: str,
        ground_truth: str,
        agent_answer: str,
) -> tuple[float, str]:
    """
    用 LLM 判断 agent 答案与 ground_truth 的吻合度，返回 (0~1 分数, 理由)。

    Args:
        judge_model: 验证/裁判模型（可与主模型相同，或换小模型省成本）
        question: 用户问题
        ground_truth: 人工标注的标准答案
        agent_answer: Agent 生成的答案

    Returns:
        (score, reason): 分数 0~1，解析失败时 score=0；reason 为模型给出的理由摘要
    """
    prompt_template = load_judge_prompts()
    prompt = prompt_template.format(
        question=question,
        ground_truth=ground_truth,
        agent_answer=agent_answer,
    )
    msg = HumanMessage(content=prompt)
    out = judge_model.invoke([msg])
    text = (out.content or "").strip()
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return 0.0, "解析失败：未找到 JSON"
    try:
        obj = json.loads(m.group())
        s = float(obj.get("score", 0))
        score = max(0.0, min(1.0, s))
        reason = str(obj.get("reason", ""))[:200]
        return score, reason
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return 0.0, f"解析失败：{e}"
