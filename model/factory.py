from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from utils.config_utils import rag_conf, api_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


def _chat_model_kwargs() -> dict:
    """DashScope Generation：temperature / max_tokens（见 config/rag.yml）。"""
    m: dict = {}
    if rag_conf.get("chat_temperature") is not None:
        m["temperature"] = float(rag_conf["chat_temperature"])
    if rag_conf.get("chat_max_tokens") is not None:
        m["max_tokens"] = int(rag_conf["chat_max_tokens"])
    return m


def default_turbo_chat_model_name() -> str:
    """轻量任务默认模型名（与主模型 rag.yml chat_model_name 分离）。"""
    return (rag_conf.get("turbo_chat_model_name") or "qwen-turbo").strip()


def make_turbo_chat_model(
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0,
) -> ChatTongyi:
    """
    构造 DashScope 轻量 ChatTongyi；model 为空时用 default_turbo_chat_model_name()。
    各场景按需传入 max_tokens / temperature（与主模型 chat_* 无关）。
    """
    name = (model or "").strip() or default_turbo_chat_model_name()
    return ChatTongyi(
        dashscope_api_key=api_conf["dashscope_api_key"],
        model=name,
        model_kwargs={
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    )


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            dashscope_api_key=api_conf["dashscope_api_key"],
            model=rag_conf["chat_model_name"],
            model_kwargs=_chat_model_kwargs(),
        )


class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            dashscope_api_key=api_conf["dashscope_api_key"],
            model=rag_conf['embedding_model_name']
        )

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingModelFactory().generator()
turbo_model = make_turbo_chat_model(max_tokens=2048, temperature=0)