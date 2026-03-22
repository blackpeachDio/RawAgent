"""
在线查询：连接已有 Chroma 索引，检索 + 总结（RAG），供 Agent 工具调用。
检索：向量粗排（fetch_k）→ 可选 BGE CrossEncoder 精排（top_n = k）→ 拼进 prompt。
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model, embedding_model
from utils.config_utils import chroma_conf
from utils.log_utils import logger
from utils.path_utils import get_abs_path
from utils.prompt_utils import load_rag_prompts


def _chroma_persist_directory() -> str:
    """与 cwd 无关，固定解析到项目内 chroma 目录（配置里 persist_directory 相对 utils）。"""
    return get_abs_path(chroma_conf["persist_directory"])


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class OnlineQueryService:
    """只读访问持久化向量库，提供检索器。"""

    def __init__(self):
        persist_dir = _chroma_persist_directory()
        logger.info("[RAG] Chroma persist_directory=%s", persist_dir)
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )

    def get_retriever(self):
        final_k = int(chroma_conf["k"])
        rerank_on = bool(chroma_conf.get("rerank_enabled", False))

        if not rerank_on:
            return self.vector_store.as_retriever(search_kwargs={"k": final_k})

        fetch_k = int(chroma_conf.get("fetch_k", max(final_k, 20)))
        if fetch_k < final_k:
            logger.warning(
                "[RAG] fetch_k(%s) < k(%s)，已把 fetch_k 调整为 k",
                fetch_k,
                final_k,
            )
            fetch_k = final_k

        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
                CrossEncoderReranker,
            )
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        except ImportError as e:
            raise ImportError(
                "已启用 BGE 精排：请安装 langchain-classic 与 sentence-transformers"
            ) from e

        base = self.vector_store.as_retriever(search_kwargs={"k": fetch_k})

        model_name = chroma_conf.get("rerank_model") or "BAAI/bge-reranker-v2-m3"
        device = str(chroma_conf.get("rerank_device", "cpu"))
        model_kwargs: dict = {"device": device}
        if chroma_conf.get("rerank_trust_remote_code"):
            model_kwargs["trust_remote_code"] = True

        cross = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
        compressor = CrossEncoderReranker(model=cross, top_n=final_k)
        logger.info(
            "[RAG] 精排: BGE CrossEncoder model=%s device=%s | 粗排 fetch_k=%s | 精排 top_n=%s",
            model_name,
            device,
            fetch_k,
            final_k,
        )

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base,
        )


class RagSummarizeService(object):
    def __init__(self):
        self._query = OnlineQueryService()
        self.retriever = self._query.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        # chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            logger.info(
                "[RAG] 用户问题: %s | 【参考资料%d】元数据: %s | 正文: %s",
                query,
                counter,
                doc.metadata,
                doc.page_content,
            )
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        if not context_docs:
            logger.warning("[RAG] 未检索到任何片段，context 为空 | 用户问题: %s", query)

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
