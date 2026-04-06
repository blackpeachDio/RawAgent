"""
hobby / character / preferences：多条事实的追加合并、注入时按日期与当前 query 相关度加权。

存储：FactualStore 中上述 key 的 value 为 JSON 数组字符串；
      每项为 { "value", "observed_at" (ISO8601 UTC), "scenario" (可选) }。
      兼容旧数据：纯字符串视为单条 legacy。
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from utils.log_utils import logger

MULTI_FACT_KEYS = frozenset({"hobby", "character", "preferences"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_value(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_multi_items(raw: str | None) -> list[dict[str, Any]]:
    """从存储值解析为条目列表；旧版纯文本 -> 单条。"""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                out: list[dict[str, Any]] = []
                for it in data:
                    if isinstance(it, dict) and (it.get("value") or "").strip():
                        out.append(
                            {
                                "value": str(it.get("value", "")).strip(),
                                "observed_at": str(it.get("observed_at") or "").strip()
                                or _utc_now_iso(),
                                "scenario": str(it.get("scenario") or "").strip(),
                            }
                        )
                return out
        except json.JSONDecodeError as e:
            logger.warning("[factual_multi] 解析多条 JSON 失败，退化为纯文本: %s", e)
    return [
        {
            "value": raw,
            "observed_at": _utc_now_iso(),
            "scenario": "",
        }
    ]


def merge_append_multi(
    existing_raw: str | None,
    new_value: str,
    scenario: str,
    *,
    max_per_key: int,
) -> str:
    """
    将新观察合并进列表：同文去重（归一化后相同则刷新 observed_at / scenario）；
    按 observed_at 新到旧截断到 max_per_key。
    """
    items = load_multi_items(existing_raw)
    nv = _normalize_value(new_value)
    if not nv:
        return json.dumps(items, ensure_ascii=False) if items else ""

    now = _utc_now_iso()
    scenario = (scenario or "").strip()

    for it in items:
        if _normalize_value(str(it.get("value", ""))) == nv:
            it["observed_at"] = now
            if scenario:
                it["scenario"] = scenario
            break
    else:
        items.append(
            {
                "value": new_value.strip(),
                "observed_at": now,
                "scenario": scenario,
            }
        )

    items.sort(key=lambda x: str(x.get("observed_at") or ""), reverse=True)
    items = items[: max(1, max_per_key)]
    return json.dumps(items, ensure_ascii=False)


def _recency_score(observed_at: str | None, conf: dict[str, Any]) -> float:
    half_life = float(conf.get("factual_multi_recency_half_life_days", 30.0) or 30.0)
    if half_life <= 0:
        half_life = 30.0
    if not observed_at:
        return 0.55
    try:
        s = str(observed_at).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return math.exp(-math.log(2) * max(0.0, days) / half_life)
    except Exception:
        return 0.5


def _relevance_score(query: str, value: str, scenario: str) -> float:
    """轻量相关度：query 与 value+scenario 的字符重叠比例 + 子串命中加成。"""
    q = (query or "").strip().lower()
    blob = f"{value} {scenario}".strip().lower()
    if not q:
        return 0.5
    if not blob:
        return 0.0
    if q in blob:
        return 1.0
    qs = set(q)
    bs = set(blob)
    inter = len(qs & bs)
    union = len(qs | bs) or 1
    return 0.35 + 0.65 * (inter / union)


def _combined_score(
    item: dict[str, Any],
    query: str,
    conf: dict[str, Any],
) -> float:
    w_r = float(conf.get("factual_multi_weight_recency", 0.35))
    w_rel = float(conf.get("factual_multi_weight_relevance", 0.65))
    if w_r + w_rel <= 0:
        w_r, w_rel = 0.35, 0.65
    s = w_r + w_rel
    w_r, w_rel = w_r / s, w_rel / s
    rec = _recency_score(str(item.get("observed_at") or ""), conf)
    rel = _relevance_score(
        query,
        str(item.get("value") or ""),
        str(item.get("scenario") or ""),
    )
    return w_r * rec + w_rel * rel


def format_factual_block_for_injection(
    factual: dict[str, str],
    query: str,
    conf: dict[str, Any],
) -> str:
    """
    生成注入用的「用户画像」正文：多值 key 按加权分排序后取 top_n 条展示。
    """
    lines: list[str] = []
    top_n = max(1, int(conf.get("factual_multi_inject_top_n", 8) or 8))

    for key in sorted(factual.keys()):
        raw = factual.get(key) or ""
        if key in MULTI_FACT_KEYS:
            items = load_multi_items(raw)
            if not items:
                continue
            scored = [( _combined_score(it, query, conf), it) for it in items]
            scored.sort(key=lambda x: x[0], reverse=True)
            picked = [x[1] for x in scored[:top_n]]
            lines.append(f"{key}:")
            for i, it in enumerate(picked, 1):
                v = str(it.get("value", "")).strip()
                sc = str(it.get("scenario") or "").strip()
                ob = str(it.get("observed_at") or "").strip()
                part = f"  [{i}] {v}"
                if sc:
                    part += f"（场景：{sc}）"
                if ob:
                    part += f" [记录: {ob[:10]}]"
                lines.append(part)
        else:
            if raw.strip():
                lines.append(f"{key}: {raw}")

    return "\n".join(lines)
