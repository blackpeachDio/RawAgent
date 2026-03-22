"""
在线查询：连接已有 Chroma 索引，检索 + 总结（RAG），供 Agent 工具调用。
检索：可选查询改写、可选 BM25 混合、向量粗排、合并去重、BGE 精排、可选分数拒答。
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from model.factory import chat_model, embedding_model
from rag.retrieval_pipeline import REFUSAL_MESSAGE, build_bm25_retriever, retrieve_documents
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
    """只读访问持久化向量库，提供检索器（统一走 retrieval_pipeline）。"""

    def __init__(self):
        persist_dir = _chroma_persist_directory()
        logger.info("[RAG] Chroma persist_directory=%s", persist_dir)
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        self._bm25_retriever = None

    def get_retriever(self):
        def _pipe(q: str) -> list[Document]:
            bm25 = None
            if chroma_conf.get("hybrid_bm25_enabled"):
                if self._bm25_retriever is None:
                    self._bm25_retriever = build_bm25_retriever(self.vector_store)
                bm25 = self._bm25_retriever
            return retrieve_documents(self.vector_store, q, bm25_retriever=bm25)

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
        context_docs = self.retriever_docs(query)

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

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
