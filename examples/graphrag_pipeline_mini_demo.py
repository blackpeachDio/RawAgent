"""
GraphRAG 四步迷你串联示例（教学用，非生产）：

1) 实体关系抽取 —— 这里用「手写三元组」模拟 LLM 输出；真实场景是 prompt 让模型输出
   (主体, 关系, 客体) 或 JSON，再做规范化与去重。
2) 图存储与查询 —— 用无向图表示实体共现/关系；查询时可做 k 跳邻域子图。
3) 社区发现与摘要 —— 用模块度贪心划分社区（需 networkx）；每个社区一段「摘要」模拟
   LLM 对社区内实体的概括；若无 networkx 则退化为「连通分量」。
4) 查询路由 —— 极简规则区分「全局/比较」vs「局部实体」，决定用社区摘要还是子图。

运行：python examples/graphrag_pipeline_mini_demo.py
可选：pip install networkx 以获得更接近论文的社区划分。
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# 1) 模拟「实体关系抽取」结果（真实管线里由 LLM + 后处理得到）
# ---------------------------------------------------------------------------
Triple = tuple[str, str, str]  # (head, relation, tail)

RAW_TRIPLES: list[Triple] = [
    ("石头T7", "采用", "LDS激光导航"),
    ("石头T7", "适合", "小户型"),
    ("石头T7", "支持", "米家App"),
    ("云鲸J3", "侧重", "拖布自清洁"),
    ("云鲸J3", "配备", "大水箱"),
    ("云鲸J3", "支持", "米家App"),
    ("LDS激光导航", "属于", "激光导航类"),
]


def build_undirected_graph(triples: list[Triple]) -> dict[str, set[str]]:
    """把三元组压成无向图：边 = 实体在句子里一起出现 / 有关系（教学用简化）。"""
    g: dict[str, set[str]] = defaultdict(set)
    for h, _rel, t in triples:
        g[h].add(t)
        g[t].add(h)
    return g


def subgraph_k_hop(g: dict[str, set[str]], center: str, k: int = 1) -> set[str]:
    """从 center 出发 k 跳内的节点（BFS）。用于「局部」检索上下文。"""
    if center not in g:
        return set()
    seen = {center}
    q = deque([(center, 0)])
    while q:
        u, d = q.popleft()
        if d >= k:
            continue
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return seen


# ---------------------------------------------------------------------------
# 2) 社区发现：优先 networkx 的 greedy_modularity_communities；否则连通分量
# ---------------------------------------------------------------------------
def communities_fallback(g: dict[str, set[str]]) -> list[set[str]]:
    """无 networkx：每个连通分量当作一个社区。"""
    visited: set[str] = set()
    out: list[set[str]] = []
    for node in g:
        if node in visited:
            continue
        comp: set[str] = set()
        stack = [node]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in g[u]:
                if v not in visited:
                    stack.append(v)
        out.append(comp)
    return out


def communities_modularity(g: dict[str, set[str]]) -> list[set[str]]:
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        return communities_fallback(g)

    G = nx.Graph()
    for u, nbrs in g.items():
        for v in nbrs:
            G.add_edge(u, v)
    comms = greedy_modularity_communities(G)
    return [set(c) for c in comms]


# ---------------------------------------------------------------------------
# 3) 模拟「每个社区的 LLM 摘要」（真实管线：把社区内三元组/文本块喂给模型生成一段）
# ---------------------------------------------------------------------------
@dataclass
class CommunityRecord:
    cid: int
    members: frozenset[str]
    summary: str


def _fake_llm_community_summary(c: set[str]) -> str:
    """按社区内实体写「伪摘要」；生产里改为：LLM(社区内三元组与原文片段) → 一段中文。"""
    if "石头T7" in c and "云鲸J3" not in c:
        return (
            "石头T7：激光导航，适合小户型，可接入米家；与导航技术类概念相关。"
        )
    if "云鲸J3" in c and "石头T7" not in c:
        return "云鲸J3：偏拖洗与自清洁、大水箱，可接入米家。"
    if "石头T7" in c and "云鲸J3" in c:
        return (
            "本社区同时包含石头与云鲸产品线，均支持米家；对比选购需看导航/拖洗侧重。"
        )
    if len(c) == 1:
        return "单点实体，信息较少。"
    return "该簇涉及多实体关系，需结合具体问题展开（此处为占位摘要）。"


def assign_summaries(comms: list[set[str]]) -> list[CommunityRecord]:
    return [
        CommunityRecord(
            cid=i,
            members=frozenset(c),
            summary=_fake_llm_community_summary(c),
        )
        for i, c in enumerate(comms)
    ]


# ---------------------------------------------------------------------------
# 4) 查询路由：局部 vs 全局（真实管线可用小分类模型或 LLM 做 intent）
# ---------------------------------------------------------------------------
Mode = Literal["local", "global"]


def route_query(q: str) -> Mode:
    q = q.strip()
    global_hints = ("哪些", "区别", "对比", "整体", "概况", "总结", "市场", "各品牌")
    local_hints = ("怎么", "什么导航", "哪款", "参数", "主刷", "水箱")
    if any(h in q for h in global_hints):
        return "global"
    if any(h in q for h in local_hints):
        return "local"
    return "local"


def retrieve_for_query(
    q: str,
    g: dict[str, set[str]],
    comm_records: list[CommunityRecord],
) -> str:
    mode = route_query(q)
    if mode == "global":
        lines = ["【路由】全局/归纳 → 使用各社区摘要："]
        for cr in comm_records:
            lines.append(f"- 社区{cr.cid}：{cr.summary}")
        return "\n".join(lines)

    # 局部：从问题里找一个锚点实体（演示：硬编码匹配）
    anchor = None
    for name in ("石头T7", "云鲸J3", "米家App", "LDS激光导航"):
        if name in q:
            anchor = name
            break
    if anchor is None:
        return "【路由】局部但未命中已知实体 → 可退回向量检索或问用户澄清。"

    nodes = subgraph_k_hop(g, anchor, k=1)
    lines = [f"【路由】局部 → 以「{anchor}」为中心 1 跳子图：", f"节点：{sorted(nodes)}"]
    # 附带相关三元组
    rels = [t for t in RAW_TRIPLES if t[0] in nodes or t[2] in nodes]
    for h, r, t in rels[:12]:
        lines.append(f"  ({h}) -{r}-> ({t})")
    return "\n".join(lines)


def main() -> None:
    g = build_undirected_graph(RAW_TRIPLES)
    comms = communities_modularity(g)
    records = assign_summaries(comms)

    print("=== 图（邻接表，节选）===")
    for k in sorted(g.keys())[:8]:
        print(f"  {k} -> {sorted(g[k])[:6]}{'...' if len(g[k]) > 6 else ''}")

    print("\n=== 社区划分 ===")
    for cr in records:
        print(f"  [{cr.cid}] {sorted(cr.members)}")
        print(f"       摘要: {cr.summary[:60]}...")

    print("\n--- 示例问 1：局部 ---")
    q1 = "石头T7 用的是什么导航？"
    print("Q:", q1)
    print(retrieve_for_query(q1, g, records))

    print("\n--- 示例问 2：全局（演示路由词）---")
    q2 = "各品牌扫地机各有什么特点？"
    print("Q:", q2)
    print(retrieve_for_query(q2, g, records))


if __name__ == "__main__":
    main()
