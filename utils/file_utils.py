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


def _word_center_in_any_table_bbox(word: dict[str, Any], bboxes: list[tuple[Any, ...]]) -> bool:
    """判断单词中心是否落在任一表格区域内（与 find_tables 的 bbox 对齐）。"""
    try:
        x0w = float(word["x0"])
        x1w = float(word["x1"])
        topw = float(word["top"])
        botw = float(word.get("bottom", word["top"]))
    except (KeyError, TypeError, ValueError):
        return False
    cx = (x0w + x1w) / 2.0
    cy = (topw + botw) / 2.0
    for box in bboxes:
        if len(box) < 4:
            continue
        bx0, btop, bx1, bbot = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if bx0 <= cx <= bx1 and btop <= cy <= bbot:
            return True
    return False


def _pdf_join_words_to_lines(words: list[dict[str, Any]]) -> str:
    """按行位置粗略拼回正文（用于表格外文本，避免与 Markdown 表重复）。"""
    if not words:
        return ""
    tol = 3.0
    sorted_w = sorted(
        words,
        key=lambda w: (round(float(w.get("top", 0)), 1), float(w.get("x0", 0))),
    )
    lines: list[list[str]] = []
    cur_top: float | None = None
    cur: list[str] = []
    for w in sorted_w:
        t = float(w.get("top", 0))
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        if cur_top is None or abs(t - cur_top) <= tol:
            cur.append(txt)
            if cur_top is None:
                cur_top = t
        else:
            if cur:
                lines.append(cur)
            cur = [txt]
            cur_top = t
    if cur:
        lines.append(cur)
    return "\n".join(" ".join(line) for line in lines)


def _pdf_page_text_outside_tables(page: Any) -> str:
    """
    页面正文：排除 find_tables 检测到的表格 bbox 内文字，仅保留「表格外」流式文本。
    与 extract_tables → Markdown 并存时，可避免同一表格在 layout 文本里再出现一遍（索引重复）。
    """
    try:
        found = page.find_tables() or []
    except Exception as e:
        logger.warning("[pdf_loader] find_tables 失败，退回全文 extract_text | %s", e)
        found = []
    bboxes = [t.bbox for t in found if getattr(t, "bbox", None)]
    if not bboxes:
        raw = page.extract_text(layout=True)
        return (raw or "").strip()

    try:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    except Exception as e:
        logger.warning("[pdf_loader] extract_words 失败，退回 extract_text | %s", e)
        raw = page.extract_text(layout=True)
        return (raw or "").strip()

    kept = [w for w in words if not _word_center_in_any_table_bbox(w, bboxes)]
    return _pdf_join_words_to_lines(kept).strip()


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
    同页内多段表合并/补头：列数一致时识别「重复表头」「续表数据」「新表（不同表头）」。
    跨页状态在 pdf_loader 中按页重置，避免分页后 ncol 相同却误用上一页表头。
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
    使用 pdfplumber 读取 PDF：正文为 **表格外** 文本（find_tables bbox 外）+ extract_tables() 转 Markdown，
    避免 layout 全文与表格单元格重复索引；表头补全仅在 **同一页内** 多段解析时生效。

    性能：大文件逐页 ``flush_cache``、可选 ``max_pages`` 上限、周期性进度日志。
    """
    import pdfplumber

    open_kw: dict[str, Any] = {}
    if passwd:
        open_kw["password"] = passwd

    documents: list[Document] = []
    table_seq = 0

    with pdfplumber.open(filepath, **open_kw) as pdf:
        total_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            # 每页独立：跨页不复用表头，避免 ncol 相同的新表误接上一页列名
            last_header: list[str] | None = None
            last_col_count = 0

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
            page_text = _pdf_page_text_outside_tables(page)
            if page_text:
                md_parts.append(page_text)

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


def md_loader(filepath: str) -> list[Document]:
    """Markdown 知识库文件：UTF-8 读入；切分由离线索引侧 TableAwareMarkdownTextSplitter 完成。"""
    docs = TextLoader(filepath, encoding="utf-8").load()
    for doc in docs:
        meta = dict(doc.metadata or {})
        meta.setdefault("loader", "markdown")
        doc.metadata = meta
    return docs


def java_loader(filepath: str) -> list[Document]:
    # Java 源码按纯文本读入，后续用代码友好 splitter 切分
    return TextLoader(filepath, encoding="utf-8").load()
