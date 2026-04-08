"""
校验 config 合并后关键键存在（agent_conf / chroma_conf）。
在项目根执行: python scripts/verify_config.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.config_utils import agent_conf, chroma_conf  # noqa: E402


def main() -> int:
    checks = [
        (agent_conf, "memory_inject_mode", "agent+memory 合并"),
        (agent_conf, "conversation_max_messages", "agent.yml"),
        (chroma_conf, "collection_name", "chroma.yml"),
        (chroma_conf, "memory_collection_name", "chroma_memory 合并"),
        (chroma_conf, "persist_directory", "chroma.yml"),
        (chroma_conf, "memory_persist_directory", "chroma_memory 合并"),
    ]
    for conf, key, hint in checks:
        if conf.get(key) in (None, ""):
            print(f"[FAIL] 缺少 {key!r} ({hint})", file=sys.stderr)
            return 1
    print("verify_config: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
