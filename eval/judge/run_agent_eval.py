"""
================================================================================
eval/run_agent_eval.py — Agent 全链路评测（含 LLM 自建打分）
================================================================================
运行方式（在项目根目录）:
    python eval/run_agent_eval.py
    python eval/run_agent_eval.py --golden eval/golden_agent.json

--------------------------------------------------------------------------------
与 run_eval.py 的区别
--------------------------------------------------------------------------------
  run_eval.py      : 仅评测 RAG（检索 + rag_summarize 生成），不经过 Agent
  run_agent_eval.py: 评测完整 Agent（含工具调用、多轮推理），用 LLM 对比 ground_truth 打分

--------------------------------------------------------------------------------
金标字段要求
--------------------------------------------------------------------------------
  question (必填)
  ground_truth_answer (必填) — 无则跳过该条（LLM 打分依赖标准答案）
  id、tier 等可选

--------------------------------------------------------------------------------
LLM 打分 vs RAG 精排打分（不是同一个东西）
--------------------------------------------------------------------------------
  RAG 精排(BGE)  : query 与 document 的语义相关性，用于检索结果重排序
  LLM 自建打分   : agent 答案与 ground_truth 的吻合度，评估整体回答质量
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

# 保证从项目根目录运行时可导入
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agent.react_agent import ReactAgent
from eval.judge.llm_judge import llm_answer_score
from model.factory import chat_model
from utils.path_utils import get_repo_root


@dataclass
class AgentGoldenItem:
    id: str
    question: str
    ground_truth_answer: str


def _load_golden(path: str) -> list[AgentGoldenItem]:
    """读取金标，仅保留有 question 和 ground_truth_answer 的条目。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[AgentGoldenItem] = []
    for row in raw:
        if not isinstance(row, dict) or "question" not in row:
            continue
        gt = (row.get("ground_truth_answer") or "").strip()
        if not gt:
            continue
        out.append(
            AgentGoldenItem(
                id=str(row.get("id", "")),
                question=str(row["question"]).strip(),
                ground_truth_answer=gt,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent 全链路评测，用 LLM 对比 ground_truth 打分。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python eval/run_agent_eval.py\n  python eval/run_agent_eval.py --golden eval/golden_agent.json",
    )
    parser.add_argument(
        "--golden",
        default=os.path.join(get_repo_root(), "eval", "judge", "golden_agent.json"),
        help="金标 JSON 路径（需含 ground_truth_answer）",
    )
    args = parser.parse_args()

    items = _load_golden(args.golden)
    if not items:
        print("金标为空，或没有同时含 question 和 ground_truth_answer 的条目，退出")
        return

    agent = ReactAgent()
    judge_model = chat_model  # 可与主模型相同；后续可从 config 读取 eval_judge_model

    scores: list[float] = []
    reasons: list[str] = []

    for it in items:
        print(f"[{it.id}] 运行 Agent: {it.question[:50]}...")
        messages = [{"role": "user", "content": it.question}]
        parts: list[str] = []
        for p in agent.execute_stream(messages):
            parts.append(p)
        agent_answer = (
            (getattr(agent, "last_turn_display_assistant_text", "") or "").strip()
            or "".join(parts).strip()
        )

        print(f"[{it.id}] Agent 回答长度: {len(agent_answer)} 字符")
        score, reason = llm_answer_score(
            judge_model,
            question=it.question,
            ground_truth=it.ground_truth_answer,
            agent_answer=agent_answer,
        )
        scores.append(score)
        reasons.append(reason)
        print(f"[{it.id}] LLM 打分={score:.3f} | {reason}")

    print("---")
    if scores:
        print(f"LLM 自建打分 (mean): {sum(scores) / len(scores):.3f} (n={len(scores)})")


if __name__ == "__main__":
    main()
