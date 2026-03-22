"""
================================================================================
eval/run_eval.py — RAG 离线评测脚本
================================================================================
运行方式（在项目根目录）:
    python eval/run_eval.py
    python eval/run_eval.py --help

--------------------------------------------------------------------------------
一、金标文件 golden_dataset.json 里每个字段是什么意思
--------------------------------------------------------------------------------
金标 = 人工事先标好的「标准」，用来和系统输出对比。JSON 是一个数组，每一项是一条样本：

  id (可选)
      字符串，方便在日志里区分第几题，不参与计算。

  question (必填)
      用户问题，会送给检索器 retriever 和（可选）RagSummarizeService。

  relevant_sources (强烈建议填)
      字符串数组，表示「这道题正确答案应该从哪些知识文件里来」。
      一般写 data 目录下文件名，如 "选购指南.txt"。
      脚本会把检索到的每条片段的 metadata.source 取 basename，与这里比对。
      为什么需要：没有它就无法自动判断「检索有没有找对文档」。

  ground_truth_answer (可选)
      人工写的标准回答。若留空，脚本不会算「答案相似度」。
      若填写，且未加 --skip-generation，会用模型生成答案后与金标算相似度。
      为什么需要：要量化「答得是否接近标准说法」时才有用。

  tier (可选)
      分层标签，如 easy、hard、regression；用于汇总时按 tier 打印 Source Hit / MRR。
      固定回归集可单独建 golden_regression.json，用 --only-tier regression 只跑回归题。

  _comment 等非标准字段
      可写在某条里作备忘；只要带 question 且结构合法即可。

--------------------------------------------------------------------------------
二、命令行参数（--xxx）分别干什么
--------------------------------------------------------------------------------
  --golden PATH
      金标 JSON 的路径，默认是与本脚本同目录的 golden_dataset.json。
      为什么：方便多套评测集（如 dev.json / hard.json）换文件对比。

  --skip-generation
      若加上：只做「检索」评测，不调用 rag_summarize（不跑大模型生成整段回答）。
      为什么：
        - 省钱、省时间；
        - 只调分片/k/embedding 时，通常先看检索是否命中，不必每次都生成。

  --faithfulness
      若加上：在「已生成回答」的前提下，再调一次 chat_model，让模型当裁判，
      给「检索上下文 vs 模型回答」打 0~1 忠实度（是否胡编）。
      为什么：量化「对上下文的幻觉」；会额外消耗 token。

--------------------------------------------------------------------------------
三、脚本里算的几个指标：含义、怎么算、为什么要算
--------------------------------------------------------------------------------
1) Source Hit@k（本脚本里打印成每条 SourceHit@k=0/1，最后汇总比例）
   - 含义：top-k 条检索结果里，是否至少有一条来自你标注的 relevant_sources（按文件名）。
   - 怎么算：对单条样本，若存在则记 1，否则 0；最后对所有带 relevant_sources 的样本取平均。
   - 为什么：检索错了后面 RAG 很难答对，这是「文档级召回」的最简代理指标。
   - 注意：k 来自 config/chroma.yml 的 k，不是命令行参数。

2) MRR_step（每条一个数，最后打印 MRR 均值）
   - 含义：第一个「相关」片段在 top-k 里排第几位。若在第一位，贡献 1；第二位，贡献 1/2；…；未命中贡献 0。
   - 公式：对单条，RR = 1/rank（rank 为首个命中位置 1..k），未命中为 0；多条样本再平均。
   - 为什么：比只看 0/1 更细——同样「命中」，排第一通常比排第三更好。

3) char_bigram_jaccard（字符二元组 Jaccard）
   - 含义：模型回答与 ground_truth_answer 有多像（形式上的重叠），范围约 0~1。
   - 怎么算：把两端文本去掉空白后，拆成相邻两个字符的集合（bigram），算
            J(A,B) = |A∩B| / |A∪B|（集合 Jaccard）。
   - 为什么：不需要额外库，中文也能用一点；但不是语义上的「准确率」，只能当粗相似度。
   - 若你要严谨语义相似度，可以后换 ROUGE/BERTScore/人工判分。

4) faithfulness（--faithfulness 时）
   - 含义：回答是否被「本次检索到的片段」支持，偏「对资料的忠实度」，不是「是否符合真实世界」。
   - 怎么算：把检索片段拼成 context，与模型回答一起发给 chat_model，解析返回 JSON 里的 score。
   - 为什么：没有人工逐句对照时，用 LLM-as-judge 是常见做法（仍有裁判模型误差）。

--------------------------------------------------------------------------------
四、代码执行顺序（主流程在 main）
--------------------------------------------------------------------------------
  加载金标 → 建 OnlineQueryService 拿 retriever（k 与 chroma 配置一致）
  → 对每条样本：retriever.invoke(question) 得到 docs
  → 用 metadata 算 Source Hit 与 RR
  → 若不 --skip-generation：RagSummarizeService.rag_summarize 得到 answer，可与金标算 Jaccard
  → 若 --faithfulness：再调 faithfulness_llm_judge
  → 打印汇总

依赖：项目已有依赖；--faithfulness 会多调大模型，产生费用。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# 保证从项目根目录以 `python eval/run_eval.py` 运行时可导入 rag、utils
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langchain_core.messages import HumanMessage

from model.factory import chat_model
from rag.online_query import OnlineQueryService, RagSummarizeService


@dataclass
class GoldenItem:
    """内存里的一条金标：与 JSON 对象对应（忽略无关字段）。"""

    id: str
    question: str
    # 标注「应该从哪些文件检索到依据」；用于文档级命中判定
    relevant_sources: list[str]
    # 可选的标准答案；有则参与 char_bigram_jaccard
    ground_truth_answer: str | None
    # 分层：easy | hard | regression 等，用于分桶统计；固定 regression 集可做 CI 回归
    tier: str = "default"


def _load_golden(path: str) -> list[GoldenItem]:
    """读取 JSON 数组，跳过缺少 question 的条目（防止误放纯说明对象）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[GoldenItem] = []
    for row in raw:
        if not isinstance(row, dict) or "question" not in row:
            continue
        g = (row.get("ground_truth_answer") or "").strip()
        tier = str(row.get("tier") or "default").strip() or "default"
        out.append(
            GoldenItem(
                id=str(row.get("id", "")),
                question=str(row["question"]).strip(),
                relevant_sources=[str(x).strip() for x in row.get("relevant_sources", []) if str(x).strip()],
                ground_truth_answer=g if g else None,
                tier=tier,
            )
        )
    return out


def _basename_from_metadata(meta: dict[str, Any]) -> str:
    """Chroma/LangChain 文档常在 metadata['source'] 存绝对路径，这里取文件名便于和金标比对。"""
    src = meta.get("source") or meta.get("file_path") or ""
    return os.path.basename(str(src))


def char_bigram_jaccard(a: str, b: str) -> float:
    """
    字符 bigram 的 Jaccard 相似度，约 0~1。
    把文本压成连续字符序列，再取所有相邻二字组构成集合 A、B，算 |A∩B|/|A∪B|。
    对中文不拆词也能粗略反映「字面重合度」；不等同于语义准确率。
    """
    a = re.sub(r"\s+", "", a)
    b = re.sub(r"\s+", "", b)
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0
    ga = {a[i : i + 2] for i in range(len(a) - 1)}
    gb = {b[i : i + 2] for i in range(len(b) - 1)}
    if not ga and not gb:
        return 1.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def source_hit_at_k(
    retrieved_metas: list[dict[str, Any]],
    expected_basenames: list[str],
) -> tuple[bool, int]:
    """
    文档级「是否命中」：top-k 里是否出现过来自金标文件的片段。

    返回:
        (是否命中, 首次命中的排位 rank，1 表示第一条就是相关文档；未命中 rank=0)
    用于后面算 MRR 的单条贡献 1/rank。
    """
    expected = {os.path.basename(x) for x in expected_basenames}
    for i, meta in enumerate(retrieved_metas):
        if _basename_from_metadata(meta) in expected:
            return True, i + 1
    return False, 0


def build_context_for_judge(docs) -> str:
    """把检索到的片段拼成一段文字，供忠实度裁判提示词使用（与 RAG 里进模型的上下文同源）。"""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"【片段{i}】{doc.page_content}")
    return "\n".join(parts)


def faithfulness_llm_judge(context: str, answer: str) -> float:
    """
    LLM-as-judge：让 chat_model 判断回答是否被 context 支持，输出 0~1。
    解析失败时返回 0.0，避免静默当成高分。
    """
    prompt = f"""你是评测员。下面「参考资料」是检索得到的片段，「助手回答」是模型根据资料生成的答案。

请判断：助手回答中是否包含参考资料**未提及或与资料矛盾**的陈述（即对资料的幻觉）。
- 1.0：所有关键陈述都能在资料中找到依据，无明显臆测
- 0.0：存在明显编造或与资料矛盾

只输出一行 JSON，不要其它文字，格式严格如下：
{{"score": <0到1之间的小数>, "reason": "<一句中文理由>"}}

【参考资料】
{context}

【助手回答】
{answer}
"""
    msg = HumanMessage(content=prompt)
    out = chat_model.invoke([msg])
    text = (out.content or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return 0.0
    try:
        obj = json.loads(m.group())
        s = float(obj.get("score", 0))
        return max(0.0, min(1.0, s))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用金标 JSON 评测检索 Hit/MRR，可选评测生成相似度与忠实度。详见本文件模块注释。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python eval/run_eval.py --skip-generation\n  python eval/run_eval.py --faithfulness",
    )
    parser.add_argument(
        "--golden",
        default=os.path.join(os.path.dirname(__file__), "golden_dataset.json"),
        help="金标 JSON 路径（默认: 与本脚本同目录的 golden_dataset.json）",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="对每条已生成回答调用 chat_model 评忠实度(0~1)，额外耗 token",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="不调用 rag_summarize，只跑检索并算 Source Hit / MRR（省生成费用）",
    )
    parser.add_argument(
        "--only-tier",
        default=None,
        help="只评测指定 tier（与金标字段 tier 一致），如 regression 用于固定回归集",
    )
    args = parser.parse_args()

    items = _load_golden(args.golden)
    if not items:
        print("金标为空或没有合法 question 条目，退出")
        return

    # OnlineQueryService：与线上一致的 Chroma + embedding + k
    oq = OnlineQueryService()
    retriever = oq.get_retriever()
    # RagSummarizeService 较重（含向量库与 chain）；仅在不跳过生成时初始化
    rag = None if args.skip_generation else RagSummarizeService()

    hits = 0
    labeled = 0
    rranks: list[float] = []
    sims: list[float] = []
    faith_scores: list[float] = []
    tier_hits: dict[str, int] = defaultdict(int)
    tier_labeled: dict[str, int] = defaultdict(int)
    tier_rranks: dict[str, list[float]] = defaultdict(list)

    for it in items:
        if args.only_tier and it.tier != args.only_tier:
            continue
        if not it.relevant_sources:
            print(f"[{it.id}] 跳过：未配置 relevant_sources，无法算检索召回代理指标")
            continue

        labeled += 1
        # docs 长度 = k（来自 chroma 配置），顺序即相似度从高到低
        docs = retriever.invoke(it.question)
        metas = [d.metadata or {} for d in docs]
        hit, pos = source_hit_at_k(metas, it.relevant_sources)
        tier_labeled[it.tier] += 1
        if hit:
            hits += 1
            tier_hits[it.tier] += 1
            # MRR 常用定义：仅关心第一个相关结果的 1/rank
            rranks.append(1.0 / pos)
            tier_rranks[it.tier].append(1.0 / pos)
        else:
            rranks.append(0.0)
            tier_rranks[it.tier].append(0.0)

        ctx = build_context_for_judge(docs)
        answer = ""
        if not args.skip_generation and rag is not None:
            answer = rag.rag_summarize(it.question).strip()
            if it.ground_truth_answer:
                sims.append(char_bigram_jaccard(answer, it.ground_truth_answer))

        if args.faithfulness and answer:
            fs = faithfulness_llm_judge(ctx, answer)
            faith_scores.append(fs)
            print(f"[{it.id}] faithfulness={fs:.3f}")

        print(
            f"[{it.id}] tier={it.tier} | SourceHit@k={'1' if hit else '0'} | "
            f"MRR_step={rranks[-1]:.3f} | "
            f"topk_sources={[ _basename_from_metadata(m) for m in metas ]}"
        )
        if answer and it.ground_truth_answer:
            print(f"[{it.id}] char_bigram_jaccard={sims[-1]:.3f}")

    print("---")
    if labeled:
        print(f"Source Hit@k (over items with labels): {hits / labeled:.3f}  (labeled={labeled})")
    else:
        print("无带 relevant_sources 的样本，未汇总 Source Hit@k")
    if rranks:
        print(f"MRR (mean reciprocal rank of first relevant): {sum(rranks) / len(rranks):.3f}")
    if sims:
        print(f"Answer similarity (char bigram Jaccard vs ground_truth, mean): {sum(sims) / len(sims):.3f}")
    if faith_scores:
        print(f"Faithfulness (LLM judge, mean): {sum(faith_scores) / len(faith_scores):.3f}")

    if tier_labeled:
        print("--- by tier ---")
        for t in sorted(tier_labeled.keys()):
            tl = tier_labeled[t]
            th = tier_hits.get(t, 0)
            tr = tier_rranks.get(t, [])
            mrr_t = sum(tr) / len(tr) if tr else 0.0
            print(
                f"tier={t} | labeled={tl} | Source Hit@k={th / tl:.3f} | MRR={mrr_t:.3f}"
            )


if __name__ == "__main__":
    main()
