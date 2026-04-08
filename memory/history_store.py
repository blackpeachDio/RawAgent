"""
完整对话历史存储（热插拔设计，便于后续切换为 MySQL 等）。

当前实现：本地文件 history/{user_id}.json
后续可替换为：MySQLHistoryStore、RedisHistoryStore 等。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from utils.path_utils import resolve_repo_path


class HistoryStore(ABC):
    """完整历史存储抽象接口，支持热插拔实现。"""

    @abstractmethod
    def append_message(
            self,
            user_id: str,
            role: str,
            content: str,
            **kwargs: Any,
    ) -> None:
        """追加一条消息。"""
        ...

    @abstractmethod
    def get_messages(
            self,
            user_id: str,
            limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """获取用户历史消息，按时间升序。limit 为 None 表示全部。"""
        ...


class FileHistoryStore(HistoryStore):
    """本地文件实现：history/{user_id}.json，每用户一个 JSON 数组。"""

    def __init__(self, base_dir: str | None = None):
        self._base_dir = resolve_repo_path(base_dir or "../history")
        os.makedirs(self._base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return os.path.join(self._base_dir, f"{safe or 'anonymous'}.json")

    def append_message(
            self,
            user_id: str,
            role: str,
            content: str,
            **kwargs: Any,
    ) -> None:
        path = self._path(user_id)
        messages = self._load(path)
        messages.append({
            "role": role,
            "content": content,
            "ts": datetime.now().isoformat(),
            **kwargs,
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def get_messages(
            self,
            user_id: str,
            limit: int | None = None,
    ) -> list[dict[str, Any]]:
        path = self._path(user_id)
        messages = self._load(path)
        if limit is not None:
            return messages[-limit:]
        return messages

    def _load(self, path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []


# 热插拔示例：后续可新增 class MySQLHistoryStore(HistoryStore) 实现相同接口
# 在 get_history_store 内切换实现即可。

_history_store: HistoryStore | None = None


def get_history_store() -> HistoryStore:
    """
    获取历史存储实例（热插拔入口，进程内单例）。
    当前为 FileHistoryStore；后续可替换为 MySQLHistoryStore 等，实现接口即可。
    """
    global _history_store
    if _history_store is None:
        _history_store = FileHistoryStore()
    return _history_store
