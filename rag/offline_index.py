"""
离线索引：从 data 目录读取文档，切分、向量化并写入 Chroma（含 MD5 去重）。

支持父子块：先按父块大小切分，再在每个父块内按子块（chunk_size）切分；
仅子块写入向量库并参与检索，元数据携带 parent_id / parent_content，在线检索后展开父块。
"""
import os
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from utils.config_utils import chroma_conf
from utils.file_utils import txt_loader, pdf_loader, get_file_md5_hex, listdir_with_allowed_type
from utils.log_utils import logger
from utils.path_utils import get_abs_path


def _split_parent_child(
        documents: list[Document],
        parent_splitter: RecursiveCharacterTextSplitter,
        child_splitter: RecursiveCharacterTextSplitter,
        parent_content_max_chars: int,
) -> list[Document]:
    """父块 → 子块；子块 page_content 用于嵌入，metadata 带完整父块供检索后展开。"""
    out: list[Document] = []
    parent_docs = parent_splitter.split_documents(documents)
    for pdoc in parent_docs:
        parent_text = pdoc.page_content or ""
        if not parent_text.strip():
            continue
        pid = str(uuid.uuid4())
        base_meta = dict(pdoc.metadata or {})
        stored = parent_text
        if parent_content_max_chars > 0 and len(stored) > parent_content_max_chars:
            stored = stored[:parent_content_max_chars] + "\n...(truncated)"
            logger.warning(
                "[离线索引] 父块过长已截断至 parent_content_max_chars=%s | source=%s",
                parent_content_max_chars,
                base_meta.get("source"),
            )
        parent_doc = Document(
            page_content=parent_text,
            metadata={
                **base_meta,
                "parent_id": pid,
                "parent_content": stored,
                "chunk_level": "child",
            },
        )
        children = child_splitter.split_documents([parent_doc])
        for j, c in enumerate(children):
            cm = dict(c.metadata or {})
            cm["child_index"] = j
            out.append(Document(page_content=c.page_content, metadata=cm))
    return out


class OfflineIndexService:
    """仅负责构建 / 增量更新向量索引，不在线回答。"""

    def __init__(self):
        persist_dir = get_abs_path(chroma_conf["persist_directory"])
        logger.info("[离线索引] Chroma persist_directory=%s", persist_dir)
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        self._parent_child = bool(chroma_conf.get("parent_child_enabled", False))
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self._parent_splitter: RecursiveCharacterTextSplitter | None = None
        if self._parent_child:
            self._parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chroma_conf.get("parent_chunk_size", 1200)),
                chunk_overlap=int(chroma_conf.get("parent_chunk_overlap", 100)),
                separators=chroma_conf["separators"],
                length_function=len,
            )
            logger.info(
                "[离线索引] 父子块已启用：父块=%s/%s 子块=%s/%s",
                chroma_conf.get("parent_chunk_size"),
                chroma_conf.get("parent_chunk_overlap"),
                chroma_conf.get("chunk_size"),
                chroma_conf.get("chunk_overlap"),
            )

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                if self._parent_child and self._parent_splitter is not None:
                    max_pc = int(chroma_conf.get("parent_content_max_chars") or 12000)
                    split_document = _split_parent_child(
                        documents,
                        self._parent_splitter,
                        self.spliter,
                        max_pc,
                    )
                else:
                    split_document = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)

                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    indexer = OfflineIndexService()
    indexer.load_document()
