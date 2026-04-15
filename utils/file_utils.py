import hashlib
import os
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from utils.log_utils import logger


def get_file_md5_hex(filepath: str):  # 获取文件的md5的十六进制字符串

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:  # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> tuple[str]:
    files = []

    # 1. 检查路径是否存在
    if not os.path.exists(path):
        logger.error(f"[listdir_with_allowed_type] 路径不存在: {path}")
        return tuple(files)  # 返回空元组，保持返回值类型一致

    # 2. 检查路径是否为文件夹
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path} 不是文件夹")
        return tuple(files)  # 修正：返回空元组而非allowed_types

    # 3. 遍历文件夹，筛选符合条件的文件（排除子文件夹）
    for f in os.listdir(path):
        file_full_path = os.path.join(path, f)
        # 确保是文件，且后缀符合要求
        if os.path.isfile(file_full_path) and f.endswith(allowed_types):
            files.append(file_full_path)

    return tuple(files)


def _format_pdf_table(table: list[list[Any]]) -> str:
    """将 pdfplumber 解析出的二维表转为按行可读的文本（单元格以 | 分隔）。"""
    if not table:
        return ""
    lines: list[str] = []
    for row in table:
        if row is None:
            continue
        cells: list[str] = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                cells.append(str(cell).replace("\n", " ").strip())
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _escape_md_table_cell(text: str) -> str:
    # Markdown 表格中 `|` 有语义，统一转义/替换；换行也压平
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def _format_pdf_table_markdown(table: list[list[Any]]) -> str:
    """将二维表转为 Markdown 表格（尽量保留可读性）。"""
    if not table:
        return ""
    # 过滤空行
    rows = [r for r in table if r is not None and any((c or "").strip() for c in r)]
    if not rows:
        return ""

    def row_cells(row: list[Any]) -> list[str]:
        out: list[str] = []
        for cell in row:
            out.append(_escape_md_table_cell("" if cell is None else str(cell)))
        return out

    header = row_cells(rows[0])
    # 如果只有一行，仍然用它做 header，避免完全无法表示
    body_rows = [row_cells(r) for r in rows[1:]] if len(rows) > 1 else []
    col_count = max(1, max(len(header), *(len(r) for r in body_rows)) if body_rows else len(header))

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (col_count - len(row))

    header = pad(header)
    body_rows = [pad(r) for r in body_rows]
    sep = ["---"] * col_count

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in body_rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """
    使用 pdfplumber 读取 PDF：逐页 extract_text(layout=True) + extract_tables()，
    **先转为 Markdown**（含表格 Markdown table），再进入后续文本切分。加密 PDF 可传 passwd。
    """
    import pdfplumber

    open_kw: dict[str, Any] = {}
    if passwd:
        open_kw["password"] = passwd

    documents: list[Document] = []
    with pdfplumber.open(filepath, **open_kw) as pdf:
        for page_index, page in enumerate(pdf.pages):
            md_parts: list[str] = [f"## Page {page_index + 1}"]
            page_text = page.extract_text(layout=True)
            if page_text and page_text.strip():
                md_parts.append(page_text.strip())

            tables = page.extract_tables() or []
            for table_index, table in enumerate(tables):
                if not table:
                    continue
                formatted = _format_pdf_table_markdown(table)
                if not formatted.strip():
                    continue
                md_parts.append(f"### Table {table_index + 1}\n\n{formatted}")

            body = "\n\n".join(md_parts).strip()
            if not body:
                continue

            documents.append(
                Document(
                    page_content=body,
                    metadata={
                        "source": filepath,
                        "page": page_index + 1,
                        "loader": "pdfplumber_markdown",
                    },
                )
            )

    return documents


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()


def java_loader(filepath: str) -> list[Document]:
    # Java 源码按纯文本读入，后续用代码友好 splitter 切分
    return TextLoader(filepath, encoding="utf-8").load()
