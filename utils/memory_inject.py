"""按配置与用户当前问句，决定是否注入事实记忆 / 向量记忆。"""
from __future__ import annotations

from typing import Any

# auto 模式下未配置关键词时使用的默认子串（任一命中则注入对应类型）
_DEFAULT_FACTUAL_KEYWORDS = [
    "用户画像",
    "偏好",
    "爱好",
    "我叫",
    "个人信息",
    "我住",
    "记住我",
    "我的情况",
]
_DEFAULT_VECTOR_KEYWORDS = [
    "上次",
    "之前",
    "记得",
    "说过",
    "历史",
    "摘要",
    "对话",
    "事件",
    "曾经",
    "以前",
    "提过",
]


def _keyword_list(conf: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    if key not in conf or conf[key] is None:
        return defaults
    return list(conf[key])


def memory_inject_flags(query: str, conf: dict[str, Any]) -> tuple[bool, bool]:
    """
    返回 (是否注入事实记忆, 是否注入向量记忆)。

    memory_inject_mode:
      - always: 两者均 True（与历史行为一致）
      - never: 两者均 False
      - auto: 根据问句是否包含对应关键词列表中的子串（任一命中即 True）
    """
    mode = (conf.get("memory_inject_mode") or "always").strip().lower()
    q = (query or "").strip()

    if mode == "never":
        return False, False
    if mode == "always":
        return True, True
    if mode != "auto":
        return True, True

    factual_kw = _keyword_list(
        conf, "memory_inject_factual_keywords", _DEFAULT_FACTUAL_KEYWORDS
    )
    vector_kw = _keyword_list(
        conf, "memory_inject_vector_keywords", _DEFAULT_VECTOR_KEYWORDS
    )

    inject_f = bool(q and any(k in q for k in factual_kw))
    inject_v = bool(q and any(k in q for k in vector_kw))
    return inject_f, inject_v
