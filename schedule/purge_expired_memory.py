"""
按 config/chroma.yml 的 memory_ttl_days，物理删除「用户向量记忆」库中过期记录。

仅操作 memory_persist_directory + memory_collection_name（与 RAG 的 chroma_db / agent 集合无关）。

用法（在项目根目录）:
  python -m schedule.purge_expired_memory

建议由系统定时任务每日执行（如 cron / Windows 任务计划程序）。
memory_ttl_days 为 0 时跳过删除（退出码 0）。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 项目根目录加入 path，便于 `python -m schedule.purge_expired_memory`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb  # noqa: E402

from utils.config_utils import chroma_conf  # noqa: E402
from utils.log_utils import logger  # noqa: E402
from utils.path_utils import get_abs_path  # noqa: E402


def _parse_created_at_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_expired(meta: dict, ttl_days: float, now: datetime) -> bool:
    if ttl_days <= 0:
        return False
    dt = _parse_created_at_utc(meta.get("created_at"))
    if dt is None:
        return False
    return (now - dt) > timedelta(days=ttl_days)


def purge_expired_memory_vectors(*, batch_size: int = 500) -> int:
    """
    扫描记忆集合，删除 created_at 早于 TTL 的条目。

    Returns:
        删除的条数。
    """
    ttl_days = float(chroma_conf.get("memory_ttl_days") or 0)
    if ttl_days <= 0:
        logger.info(
            "[purge_memory] memory_ttl_days=%s，未启用 TTL，跳过清理",
            chroma_conf.get("memory_ttl_days"),
        )
        return 0

    persist = get_abs_path(chroma_conf.get("memory_persist_directory", "../chroma_memory_db"))
    name = chroma_conf.get("memory_collection_name", "user_memory")

    client = chromadb.PersistentClient(path=persist)
    try:
        coll = client.get_collection(name=name)
    except Exception as e:
        logger.warning("[purge_memory] 无法打开集合 %s（可能尚未创建）: %s", name, e)
        return 0

    now = datetime.now(timezone.utc)
    expired_ids: list[str] = []
    offset = 0

    while True:
        res = coll.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        if not ids:
            break

        for i, doc_id in enumerate(ids):
            meta = dict(metas[i] if i < len(metas) else {}) or {}
            if _is_expired(meta, ttl_days, now):
                expired_ids.append(doc_id)

        offset += len(ids)
        if len(ids) < batch_size:
            break

    if not expired_ids:
        logger.info("[purge_memory] 无过期记录需删除")
        return 0

    deleted_total = 0
    for i in range(0, len(expired_ids), batch_size):
        chunk = expired_ids[i : i + batch_size]
        coll.delete(ids=chunk)
        deleted_total += len(chunk)
        logger.info("[purge_memory] 已删除 %s 条（累计 %s）", len(chunk), deleted_total)

    logger.info("[purge_memory] 完成，共删除 %s 条过期记忆", deleted_total)
    return deleted_total


def main() -> None:
    purge_expired_memory_vectors()


if __name__ == "__main__":
    main()
