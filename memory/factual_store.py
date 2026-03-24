"""
事实性记忆存储（热插拔）：hobby、name、偏好等，必须覆盖更新不能追加。

实现：InMemoryFactualStore（当前）、RedisFactualStore（后期）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from utils.config_utils import agent_conf
from utils.log_utils import logger


class FactualStore(ABC):
    """事实性记忆抽象：key-value 覆盖更新。"""

    @abstractmethod
    def set(self, user_id: str, key: str, value: str) -> None:
        """设置事实，同 key 覆盖旧值。"""

    @abstractmethod
    def get_all(self, user_id: str) -> dict[str, str]:
        """获取用户全部事实，返回 {key: value}。"""

    def delete(self, user_id: str, key: str) -> None:
        """删除指定 key（可选实现）。"""
        raise NotImplementedError


class InMemoryFactualStore(FactualStore):
    """内存实现：进程内 dict，重启丢失。后期可替换为 RedisFactualStore。"""

    def __init__(self):
        # user_id -> {key -> value}
        self._data: dict[str, dict[str, str]] = {}
        logger.info("[FactualStore] 使用 InMemoryFactualStore")

    def set(self, user_id: str, key: str, value: str) -> None:
        self._data.setdefault(user_id, {})[key] = value

    def get_all(self, user_id: str) -> dict[str, str]:
        return dict(self._data.get(user_id, {}))


class RedisFactualStore(FactualStore):
    """Redis 实现（占位）：后期接入 Redis 持久化，替换内存。"""

    def __init__(self):
        # TODO: self._redis = redis.Redis(...)
        raise NotImplementedError("RedisFactualStore 未实现，请使用 factual_store_type: memory")

    def set(self, user_id: str, key: str, value: str) -> None:
        raise NotImplementedError

    def get_all(self, user_id: str) -> dict[str, str]:
        raise NotImplementedError


_factual_store: FactualStore | None = None


def get_factual_store() -> FactualStore:
    """获取事实性记忆存储实例（按配置热插拔，单例）。"""
    global _factual_store
    if _factual_store is None:
        store_type = (agent_conf.get("factual_store_type") or "memory").strip().lower()
        if store_type == "memory":
            _factual_store = InMemoryFactualStore()
        elif store_type == "redis":
            raise NotImplementedError("RedisFactualStore 未实现，请将 factual_store_type 设为 memory")
        else:
            logger.warning("[FactualStore] 未知类型 %s，使用 InMemoryFactualStore", store_type)
            _factual_store = InMemoryFactualStore()
    return _factual_store
