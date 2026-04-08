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
from utils.path_utils import get_config_path, get_repo_root, resolve_repo_path  # noqa: E402


def main() -> int:
    # 路径解析与 get_config_path / 仓库根一致
    agent_yml = get_config_path("agent.yml")
    if os.path.normpath(resolve_repo_path("../config/agent.yml")) != os.path.normpath(agent_yml):
        print("[FAIL] resolve_repo_path(../config/agent.yml) != get_config_path(agent.yml)", file=sys.stderr)
        return 1
    hist = os.path.join(get_repo_root(), "history")
    if os.path.normpath(resolve_repo_path("../history")) != os.path.normpath(hist):
        print("[FAIL] resolve_repo_path(../history) 与仓库根/history 不一致", file=sys.stderr)
        return 1

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
