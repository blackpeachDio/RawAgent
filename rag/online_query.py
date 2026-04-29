"""
在线查询：连接已有向量索引（Chroma/Milvus 按 ``chroma_conf.vector_backend`` 切换），检索 + 总结（RAG），供 Agent 工具调用。
检索：可选查询改写、可选 BM25+向量 Ensemble（RRF）、向量粗排、合并去重、百炼 qwen3-rerank 精排、可选分数拒答。
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from model.factory import chat_model, embedding_model
from rag.retrieval_pipeline import (
    REFUSAL_MESSAGE,
    build_bm25_retriever,
    retrieve_documents,
    warmup_vector_retrieval,
)
from rag.vector_backend import make_rag_vector_backend
from utils.config_utils import chroma_conf
from utils.log_utils import log_timing, logger
from utils.path_utils import resolve_repo_path
from utils.prompt_utils import load_rag_prompts
from utils.prompt_log_utils import get_prompt_log_config, maybe_truncate, log_truncated_block


def _chroma_persist_directory() -> str:
    """与 cwd 无关，固定解析到仓库根下向量目录（Chroma 用；Milvus 不读该值）。"""
    return resolve_repo_path(chroma_conf["persist_directory"])


class OnlineQueryService:
    """只读访问持久化向量库，提供检索器（统一走 retrieval_pipeline + vector_backend）。"""

    def __init__(self):
        persist_dir = _chroma_persist_directory()
        self.backend = make_rag_vector_backend(
            embedding_model,
            collection_name=chroma_conf["collection_name"],
            persist_directory=persist_dir,
        )
        self._bm25_retriever = None
        warmup_vector_retrieval(self.backend)

    def get_retriever(self):
        def _pipe(q: str) -> list[Document]:
            bm25 = None
            if chroma_conf.get("hybrid_bm25_enabled"):
                if self._bm25_retriever is None:
                    self._bm25_retriever = build_bm25_retriever(self.backend)
                bm25 = self._bm25_retriever
            return retrieve_documents(self.backend, q, bm25_retriever=bm25)

        return RunnableLambda(_pipe)


class RagSummarizeService(object):
    def __init__(self):
        self._query = OnlineQueryService()
        self.retriever = self._query.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        log_timing("rag_summarize", "retrieve_start")
        context_docs = self.retriever_docs(query)
        log_timing("rag_summarize", "retrieve_done")

        if not context_docs:
            logger.warning("[RAG] 无可用检索片段或已触发拒答阈值 | 用户问题: %s", query)
            return REFUSAL_MESSAGE

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

        try:
            # 打印“提交给大模型”的 rag_summarize prompt 细节（便于排查/对齐）。
            full, max_chars = get_prompt_log_config()
            _maybe_truncate = lambda s: maybe_truncate(s, full=full, max_chars=max_chars)

        except Exception as e:
            logger.warning("[RAG] 打印 prompt 失败：%s", str(e))

        log_timing("rag_summarize", "summarize_llm_start")
        result = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        log_timing("rag_summarize", "summarize_llm_done")

        # 打印 rag_summarize 最终总结结果，便于排查“模型是怎么基于参考资料输出的”
        try:
            full, max_chars = get_prompt_log_config()
            truncated = maybe_truncate(str(result), full=full, max_chars=max_chars)
            log_truncated_block(
                logger,
                "[RAG_RESULT_BEGIN]",
                "[RAG_RESULT_END]",
                truncated,
            )
        except Exception as e:
            logger.debug("[RAG] 打印结果失败：%s", str(e))

        return result


if __name__ == "__main__":
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
