"""
PDF 转 Markdown 后的切分：普通段落走 MarkdownTextSplitter；管道表格按行切块，
并在每块前重复「紧邻表格的 ### 表名 + 列头」，避免检索丢失表级与列级语义。
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_text_splitters import MarkdownTextSplitter

def _detach_trailing_heading_before_table(buf: list[str]) -> tuple[list[str], str | None]:
    """
    取紧邻管道表之前的 Markdown 标题（如 ### Table 1）作为表名，从正文缓冲中移除，
    避免仅出现在首块；后续由表分块逻辑在每片前重复。
    """
    if not buf:
        return buf, None
    i = len(buf) - 1
    while i >= 0 and buf[i].strip() == "":
        i -= 1
    if i < 0:
        return buf, None
    line = buf[i]
    # 仅 ###～######，避免把「## Page N」当成表名；与 pdf_loader 的 ### Table k 一致
    if not re.match(r"^\s*#{3,6}\s+\S", line):
        return buf, None
    caption = line.strip()
    new_buf = buf[:i]
    while new_buf and new_buf[-1].strip() == "":
        new_buf.pop()
    return new_buf, caption


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False
    cells = [c.strip() for c in s.split("|") if c.strip() != ""]
    if not cells:
        return False
    for c in cells:
        if not re.match(r"^:?-{3,}:?$", c):
            return False
    return True


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _iter_md_pipe_table_segments(
        text: str,
) -> list[tuple[Literal["text", "table"], str, str | None]]:
    """将全文拆成交替的 text / table 段；table 段可带表名（紧邻表格的 Markdown 标题）。"""
    lines = text.split("\n")
    out: list[tuple[Literal["text", "table"], str, str | None]] = []
    i = 0
    n = len(lines)

    def flush_text(buf: list[str]) -> None:
        block = "\n".join(buf).strip("\n")
        if block.strip():
            out.append(("text", block, None))

    text_buf: list[str] = []
    while i < n:
        if (
            i + 1 < n
            and _is_pipe_row(lines[i])
            and _is_separator_row(lines[i + 1])
        ):
            text_buf, caption = _detach_trailing_heading_before_table(text_buf)
            flush_text(text_buf)
            text_buf = []
            j = i + 2
            while j < n and _is_pipe_row(lines[j]):
                j += 1
            table_block = "\n".join(lines[i:j]).strip()
            if table_block:
                out.append(("table", table_block, caption))
            i = j
            continue
        text_buf.append(lines[i])
        i += 1

    flush_text(text_buf)
    return out


def _split_table_repeat_header(
        table_md: str,
        chunk_size: int,
        caption: str | None,
) -> list[str]:
    """大表格按行切块；每块含可选表名 + 表头 + 分隔行 + 若干数据行。"""
    cap = (caption.strip() + "\n\n") if caption and caption.strip() else ""
    cap_len = len(cap)

    lines = table_md.strip().split("\n")
    if len(lines) < 2:
        one = cap + table_md if table_md.strip() else cap.rstrip()
        if not one.strip():
            return []
        return [one] if len(one) <= chunk_size else []

    header = lines[0]
    sep = lines[1]
    body = lines[2:]
    core_prefix = header + "\n" + sep
    prefix = cap + core_prefix
    prefix_len = cap_len + len(core_prefix) + 1  # newline before first body row

    full_one = cap + table_md
    if len(full_one) <= chunk_size:
        return [full_one]

    if not body:
        return [full_one]

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = prefix_len

    for row in body:
        row_len = len(row) + 1
        if cur and cur_len + row_len > chunk_size:
            chunks.append(prefix + "\n" + "\n".join(cur))
            cur = [row]
            cur_len = prefix_len + row_len
        else:
            cur.append(row)
            cur_len += row_len

    if cur:
        chunks.append(prefix + "\n" + "\n".join(cur))

    return chunks if chunks else [full_one]


class TableAwareMarkdownTextSplitter(MarkdownTextSplitter):
    """
    非表格：父类 MarkdownTextSplitter（标题级切分与 overlap）。
    表格：不跨表切断；超长表格按行切且每片重复「表名（### …）+ 表头」。
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        **kwargs,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._chunk_size = int(chunk_size)
        self._md = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        text = text or ""
        segments = _iter_md_pipe_table_segments(text)
        if not segments:
            return self._md.split_text(text)

        parts: list[str] = []
        for kind, block, caption in segments:
            if kind == "text":
                parts.extend(self._md.split_text(block))
            else:
                parts.extend(
                    _split_table_repeat_header(block, self._chunk_size, caption),
                )

        return [p for p in parts if p.strip()]
