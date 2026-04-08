"""
记忆抽取任务队列：与 UI 解耦，由 Agent 后端在对话结束后入队，后台线程或独立进程消费。

- memory_queue_mode=thread（默认）：进程内 queue.Queue + 守护线程调用 extract_and_store。
- memory_queue_mode=file：仅将任务 JSON 写入 memory_queue_dir/pending/，由另起进程运行
  ``python -m memory.queue_consumer`` 消费（适合与 Streamlit 进程分离）。
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path

_q: queue.Queue[dict[str, Any]] | None = None
_worker_lock = threading.Lock()
_worker_started = False


def _pending_dir() -> Path:
    rel = (agent_conf.get("memory_queue_dir") or "../data/memory_queue").strip()
    return Path(resolve_repo_path(rel)) / "pending"


def enqueue_memory_job(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    非阻塞：将一轮「用户问 + 助手全文」写入队列（或落盘），由消费者做记忆抽取。
    """
    uid = (user_id or "").strip()
    if not uid:
        return
    am = (assistant_msg or "").strip()
    if not am:
        return

    payload = {
        "user_id": uid,
        "user_msg": (user_msg or "").strip(),
        "assistant_msg": am,
    }
    mode = (agent_conf.get("memory_queue_mode") or "thread").strip().lower()

    if mode == "file":
        _enqueue_file(payload)
        logger.debug("[memory_queue] 已写入文件队列 user_id=%s", uid)
        return

    if mode != "thread":
        logger.warning("[memory_queue] 未知 memory_queue_mode=%s，按 thread 处理", mode)

    _ensure_thread_worker()
    assert _q is not None
    _q.put(payload)
    logger.debug("[memory_queue] 已入内存队列 user_id=%s", uid)


def _enqueue_file(payload: dict[str, Any]) -> None:
    d = _pending_dir()
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"{uuid.uuid4().hex}.json"
    tmp = d / f".{fn.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(fn)


def _worker_loop() -> None:
    from memory.extract_store import extract_and_store

    assert _q is not None
    while True:
        try:
            item = _q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            extract_and_store(
                item["user_id"],
                item["user_msg"],
                item["assistant_msg"],
            )
        except Exception as e:
            logger.warning("[memory_queue] 抽取失败: %s", e, exc_info=True)
        finally:
            _q.task_done()


def _ensure_thread_worker() -> None:
    global _q, _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _q = queue.Queue()
        t = threading.Thread(target=_worker_loop, name="memory-queue-worker", daemon=True)
        t.start()
        _worker_started = True
        logger.info("[memory_queue] 守护线程已启动（thread 模式，消费记忆抽取任务）")


def ensure_memory_worker_for_tests() -> None:
    """测试用：确保 thread 模式下线程已起来。"""
    if (agent_conf.get("memory_queue_mode") or "thread").strip().lower() == "thread":
        _ensure_thread_worker()
