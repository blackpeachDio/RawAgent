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