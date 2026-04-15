"""父子块：父块正文存 SQLite 映射表，Chroma 仅存子块向量 + 轻量 metadata（parent_id 等）。"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from utils.config_utils import chroma_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path

_lock = threading.Lock()
_store = None  # ParentContentStore | None


class ParentContentStore:
    """parent_id -> 父块全文；与向量库 collection 并列，由配置路径指定。"""

    def __init__(self, db_path: str):
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, check_same_thread=False)

    def _init_schema(self) -> None:
        with self._connect() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunk (
                    parent_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT
                )
                """
            )

    def put(self, parent_id: str, content: str, source: str | None = None) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO parent_chunk (parent_id, content, source)
                VALUES (?, ?, ?)
                """,
                (parent_id, content, source),
            )

    def get(self, parent_id: str) -> str | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT content FROM parent_chunk WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
        return row[0] if row else None

    def get_many(self, parent_ids: list[str]) -> dict[str, str]:
        if not parent_ids:
            return {}
        uniq: list[str] = list(dict.fromkeys(parent_ids))
        placeholders = ",".join("?" * len(uniq))
        with self._connect() as c:
            rows = c.execute(
                f"SELECT parent_id, content FROM parent_chunk WHERE parent_id IN ({placeholders})",
                uniq,
            ).fetchall()
        return {pid: text for pid, text in rows}


def get_parent_store() -> ParentContentStore | None:
    """配置了 parent_child_map_path 时返回单例；否则为 None（有 parent_id 也无法查父块，展开时保留子块）。"""
    global _store
    raw = chroma_conf.get("parent_child_map_path")
    if not raw or not str(raw).strip():
        return None
    with _lock:
        if _store is None:
            path = resolve_repo_path(str(raw).strip())
            _store = ParentContentStore(path)
            logger.info("[RAG] 父子映射表 path=%s", path)
        return _store


def reset_parent_store_singleton() -> None:
    """测试或热切换配置时可调用。"""
    global _store
    with _lock:
        _store = None
