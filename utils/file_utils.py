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


def _format_escaped_md_table(header: list[str], body_rows: list[list[str]]) -> str:
    """已由 pdf 单元格转义过的表头/表体，拼成 Markdown pipe table。"""
    if not header and not body_rows:
        return ""
    col_count = max(
        1,
        len(header),
        max((len(r) for r in body_rows), default=0),
    )

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


def _pdf_row_cells(row: list[Any]) -> list[str]:
    out: list[str] = []
    for cell in row:
        out.append(_escape_md_table_cell("" if cell is None else str(cell)))
    return out


def _normalize_pdf_table_rows(table: list[list[Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for r in table:
        if r is None:
            continue
        if not any((c or "").strip() for c in r):
            continue
        rows.append(_pdf_row_cells(r))
    return rows


def _row_key_tuple(row: list[str]) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in row)


def _rows_similar_pdf(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return _row_key_tuple(a) == _row_key_tuple(b)


def _mostly_numeric_token(s: str) -> bool:
    s = (s or "").replace(",", "").replace("%", "").replace(" ", "").strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _looks_like_header_row_cells(cells: list[str]) -> bool:
    """启发式：多数单元格非「纯数值」则更像表头。"""
    if not cells:
        return False
    numish = sum(1 for c in cells if _mostly_numeric_token(c))
    return numish < max(1, (len(cells) + 1) // 2)


def _pad_row_to(row: list[str], n: int) -> list[str]:
    return row + [""] * (n - len(row))


def _format_pdf_table_markdown(table: list[list[Any]]) -> str:
    """将二维表转为 Markdown 表格（尽量保留可读性）。"""
    if not table:
        return ""
    rows = [r for r in table if r is not None and any((c or "").strip() for c in r)]
    if not rows:
        return ""
    header = _pdf_row_cells(rows[0])
    body_rows = [_pdf_row_cells(r) for r in rows[1:]] if len(rows) > 1 else []
    return _format_escaped_md_table(header, body_rows)


def _pdf_logical_header_body(
        rows: list[list[str]],
        last_header: list[str] | None,
        last_col_count: int,
) -> tuple[list[str], list[list[str]], list[str] | None, int]:
    """
    跨页表合并/补头：在列数一致时识别「重复表头」「续表数据」「新表（不同表头）」。
    返回 (header_cells, body_rows, new_last_header, new_last_col_count)。
    """
    ncol = max((len(r) for r in rows), default=0)
    if ncol <= 0:
        return [], [], last_header, last_col_count
    rows = [_pad_row_to(r, ncol) for r in rows]

    if last_header is None or ncol != last_col_count:
        header_cells = rows[0]
        body = rows[1:]
        return header_cells, body, header_cells, ncol

    lh = _pad_row_to(last_header, ncol)
    r0 = rows[0]
    if _rows_similar_pdf(r0, lh):
        return lh, rows[1:], last_header, last_col_count
    if _looks_like_header_row_cells(r0) and not _rows_similar_pdf(r0, lh):
        return r0, rows[1:], r0, ncol
    return lh, rows, last_header, last_col_count


def pdf_loader(
        filepath: str,
        passwd=None,
        *,
        max_pages: int = 0,
        log_progress_every: int = 25,
) -> list[Document]:
    """
    使用 pdfplumber 读取 PDF：逐页 extract_text(layout=True) + extract_tables()，
    先转为 Markdown（含表格）；**跨页续表**时复用上一页的表头行，避免第二页缺列名。

    性能：大文件逐页 ``flush_cache``、可选 ``max_pages`` 上限、周期性进度日志。
    """
    import pdfplumber

    open_kw: dict[str, Any] = {}
    if passwd:
        open_kw["password"] = passwd

    documents: list[Document] = []
    last_header: list[str] | None = None
    last_col_count = 0
    table_seq = 0

    with pdfplumber.open(filepath, **open_kw) as pdf:
        total_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            if max_pages > 0 and page_index >= max_pages:
                logger.warning(
                    "[pdf_loader] 已达 max_pages=%s，提前结束 | pages=%s | %s",
                    max_pages,
                    total_pages,
                    filepath,
                )
                break

            if log_progress_every > 0 and (page_index + 1) % log_progress_every == 0:
                logger.info(
                    "[pdf_loader] 进度 %s/%s | %s",
                    page_index + 1,
                    total_pages,
                    filepath,
                )

            md_parts: list[str] = [f"## Page {page_index + 1}"]
            page_text = page.extract_text(layout=True)
            if page_text and page_text.strip():
                md_parts.append(page_text.strip())

            tables = page.extract_tables() or []
            for table in tables:
                if not table:
                    continue
                rows_raw = _normalize_pdf_table_rows(table)
                if not rows_raw:
                    continue
                header_cells, body_rows, last_header, last_col_count = _pdf_logical_header_body(
                    rows_raw,
                    last_header,
                    last_col_count,
                )
                formatted = _format_escaped_md_table(header_cells, body_rows)
                if not formatted.strip():
                    continue
                table_seq += 1
                md_parts.append(f"### Table {table_seq}\n\n{formatted}")

            body = "\n\n".join(md_parts).strip()
            if not body:
                try:
                    page.flush_cache()
                except Exception:
                    pass
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

            try:
                page.flush_cache()
            except Exception:
                pass

    return documents


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()


def java_loader(filepath: str) -> list[Document]:
    # Java 源码按纯文本读入，后续用代码友好 splitter 切分
    return TextLoader(filepath, encoding="utf-8").load()
