"""
轻量路径自检：仅依赖 path_utils，不加载 config_utils（避免触发 api.yml / 密钥校验）。
项目根执行: python scripts/verify_paths.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.path_utils import get_config_path, get_repo_root, resolve_repo_path  # noqa: E402


def main() -> int:
    agent_yml = get_config_path("agent.yml")
    if os.path.normpath(resolve_repo_path("../config/agent.yml")) != os.path.normpath(agent_yml):
        print("[FAIL] resolve_repo_path(../config/agent.yml) != get_config_path(agent.yml)", file=sys.stderr)
        return 1
    hist = os.path.join(get_repo_root(), "history")
    if os.path.normpath(resolve_repo_path("../history")) != os.path.normpath(hist):
        print("[FAIL] resolve_repo_path(../history) 与仓库根/history 不一致", file=sys.stderr)
        return 1
    print("verify_paths: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
