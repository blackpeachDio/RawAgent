"""
使用 RAGAS 对当前项目的「检索 + rag_summarize 生成」链路做离线评测。

依赖（建议在 Python 3.11 / 3.12 下安装；3.14 可能缺少部分库的预编译包）::
    pip install ragas datasets

---------------------------------------------------------------------------
一、整体在干什么（流水线）
---------------------------------------------------------------------------
1) 读取金标 JSON：每条含用户问题 question、标准答案 ground_truth（人工编写，代表「理想回答应覆盖的要点」）。
2) 对每条问题调用与本线上一致的 RAG：OnlineQueryService 检索 + RagSummarize 生成答案。
3) 组装成 HuggingFace datasets.Dataset，列包括：
   - question：用户问题
   - answer：RAG 实际生成的回答
   - contexts：检索到的文档正文列表（每条样本一个 list[str]）
   - ground_truth：金标答案（RAGAS 中部分指标需要）
4) 调用 ragas.evaluate()，用评测 LLM + 向量模型（与业务共用 DashScope 时可与 chat_model/embedding 一致）
   对各条样本计算选定的指标，得到每条分数及汇总。

---------------------------------------------------------------------------
二、各指标在量化什么（含义与常见范围）
---------------------------------------------------------------------------
以下分数一般为 0~1，越高越好（具体实现以 RAGAS 版本文档为准）。

faithfulness（忠实度 / 幻觉风险）
    衡量：生成答案中的论断，有多少能从「给定 contexts」中推断或支撑。
    量化：基于 LLM 将答案拆成陈述并逐条对照上下文做判定，再聚合为标量。
    低分含义：模型在胡编、或过度发挥，未严格依据检索片段。

answer_relevancy（答案相关性）
    衡量：生成答案与「用户问题」的语义相关程度。
    量化：常用「逆问题」思路：据答案生成伪问题，再与原始问题的嵌入相似度等。
    低分含义：答非所问、绕题、泛泛而谈。

context_precision（上下文精确率）
    衡量：检索到的片段里，有多少是「对回答该问题真正有用」的；强调排序靠前是否更相关。
    量化：结合 question、contexts、ground_truth（或 answer）做判定。
    低分含义：检索进很多无关 chunk、或相关段排在后面。

context_recall（上下文召回率）
    衡量：金标答案中的信息，有多少被「检索到的 contexts」所覆盖。
    量化：将 ground_truth 拆成陈述，看每条是否能在 contexts 中找到依据。
    低分含义：检索漏掉了回答所必需的知识（即使生成模型很好也答不全）。

---------------------------------------------------------------------------
三、拿到分数后如何调优（对应关系）
---------------------------------------------------------------------------
- faithfulness 低、context_precision 尚可：加强 prompts/rag_summarize.txt 里「严禁编造、仅依据资料」；
  或降低 temperature；检查是否把无关片段拼进 context。
- answer_relevancy 低：检查问题是否过短/歧义；检索 query 是否需改写（query_rewrite）；总结提示是否要求紧扣问题。
- context_precision 低：chunk 过大/过小、混合检索、精排（BGE）、fetch_k/k、去重策略；考虑清洗索引噪声。
- context_recall 低：embedding 模型与领域是否匹配；是否需关键词/BM25 混合；扩充语料或调整切分。
- 多指标同时差：优先看 recall（有没有找对材料），再看 faithfulness（有没有瞎编），最后看 relevancy（是否答到点子上）。

---------------------------------------------------------------------------
四、与本项目其它评测的区别
---------------------------------------------------------------------------
- eval/llm_judge.py：用单一裁判模型比「整段答案 vs 金标」，偏端到端主观分。
- RAGAS：多维度、可分解到「检索 vs 生成」，更适合迭代 RAG 管线。

运行示例（在项目根目录，且已配置 api.yml / 建好 Chroma 索引）::

    python -m eval.ragas_rag_eval

或复制 data/rag_eval_gold.json 为 data/rag_eval_gold.json 后修改金标再跑。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 项目根：保证可 python -m eval.ragas_rag_eval
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_eval_config() -> dict[str, Any]:
    import os
    import yaml

    from utils.path_utils import get_repo_root

    p = os.path.join(get_repo_root(), "config", "eval_rag.yml")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _load_gold(path: str) -> list[dict[str, str]]:
    from utils.path_utils import get_abs_path

    fp = get_abs_path(path)
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("金标文件须为 JSON 数组")
    for i, row in enumerate(data):
        if "question" not in row or "ground_truth" not in row:
            raise ValueError(f"第 {i} 条须含 question 与 ground_truth")
    return data


def _wrap_ragas_models():
    """将 LangChain 的 Chat / Embeddings 交给 RAGAS（不同版本类名可能不同）。"""
    from model.factory import chat_model, embedding_model

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(embedding_model)
    except ImportError:
        pass
    try:
        from ragas.llms.langchain import LangchainLLM
        from ragas.embeddings.langchain import LangchainEmbeddings

        return LangchainLLM(chat_model), LangchainEmbeddings(embedding_model)
    except ImportError as e:
        raise ImportError(
            "未找到 RAGAS 的 LangChain 适配类。请安装: pip install 'ragas>=0.1.0' datasets\n"
            "Windows / Python 3.14 若编译失败，请使用 Python 3.11 或 3.12 新建 venv 再安装。"
        ) from e


def _default_metric_objects():
    """解析 config 中的指标名 -> RAGAS 指标对象。"""
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def _run_rag_once(question: str) -> tuple[str, list[str]]:
    """
    跑一条与本项目线上一致的 RAG。
    返回 (生成答案, 检索片段正文列表)。
    """
    from rag.online_query import RagSummarizeService

    svc = RagSummarizeService()
    docs = svc.retriever_docs(question)
    contexts = [d.page_content or "" for d in docs]
    answer = svc.rag_summarize(question)
    return answer, contexts


def build_evaluation_dataset(gold_rows: list[dict[str, str]]) -> Any:
    """组装 RAGAS 所需的 HuggingFace Dataset（列名与 RAGAS 约定一致）。"""
    from datasets import Dataset

    questions: list[str] = []
    answers: list[str] = []
    contexts_col: list[list[str]] = []
    ground_truths: list[str] = []

    for row in gold_rows:
        q = (row.get("question") or "").strip()
        gt = (row.get("ground_truth") or "").strip()
        ans, ctxs = _run_rag_once(q)
        questions.append(q)
        answers.append(ans)
        contexts_col.append(ctxs)
        ground_truths.append(gt)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_col,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation() -> Any:
    """加载配置 → 金标 → 跑 RAG → RAGAS evaluate → 返回结果对象。"""
    from ragas import evaluate

    cfg = _load_eval_config()
    gold_path = cfg.get("gold_dataset_path", "../data/rag_eval_gold.json")
    gold_rows = _load_gold(gold_path)

    ds = build_evaluation_dataset(gold_rows)
    llm, embeddings = _wrap_ragas_models()

    reg = _default_metric_objects()
    names = cfg.get("metrics") or list(reg.keys())
    metrics = [reg[k] for k in names if k in reg]
    if not metrics:
        metrics = list(reg.values())

    # RAGAS：在数据集上批量算分（内部会多次调用评测 LLM / 嵌入）
    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    return result


def main() -> None:
    try:
        result = run_evaluation()
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(result)
    if hasattr(result, "to_pandas"):
        print(result.to_pandas())
    elif hasattr(result, "scores"):
        print("scores:", result.scores)


if __name__ == "__main__":
    main()
