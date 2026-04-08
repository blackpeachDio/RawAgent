"""
独立进程：消费 file 模式写入的 pending/*.json，执行记忆抽取。

运行（项目根目录）::

    python -m memory.queue_consumer

配置：agent.yml 中 memory_queue_mode=file，且 memory_queue_dir 与写入方一致。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path


def _pending_dir() -> Path:
    rel = (agent_conf.get("memory_queue_dir") or "../data/memory_queue").strip()
    return Path(resolve_repo_path(rel)) / "pending"


def _process_one(path: Path) -> None:
    from memory.extract_store import extract_and_store

    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    uid = str(payload.get("user_id") or "").strip()
    if not uid:
        path.unlink(missing_ok=True)
        return
    extract_and_store(
        uid,
        str(payload.get("user_msg") or ""),
        str(payload.get("assistant_msg") or ""),
    )
    path.unlink(missing_ok=True)
    logger.info("[memory_queue] 已处理并删除 %s user_id=%s", path.name, uid)


def run_loop(*, poll_s: float = 0.5) -> None:
    pending = _pending_dir()
    pending.mkdir(parents=True, exist_ok=True)
    logger.info("[memory_queue] 文件消费者监听目录: %s", pending)
    while True:
        try:
            for path in sorted(pending.glob("*.json")):
                try:
                    _process_one(path)
                except Exception as e:
                    logger.warning("[memory_queue] 处理 %s 失败: %s", path, e, exc_info=True)
        except KeyboardInterrupt:
            logger.info("[memory_queue] 消费者退出")
            sys.exit(0)
        time.sleep(poll_s)


if __name__ == "__main__":
    run_loop()
