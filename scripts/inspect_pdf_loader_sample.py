"""
检查 pdf_loader：表格外正文 vs 全文 layout 长度、表格数量，用于发现「正文+Markdown 表」重复。
用法（在项目根、已安装 pdfplumber）::
    python scripts/inspect_pdf_loader_sample.py [path/to.pdf]
默认: data/sample_with_tables.pdf
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.file_utils import pdf_loader  # noqa: E402


def main() -> None:
    path = os.path.join(_REPO, "data", "sample_with_tables.pdf")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if not os.path.isfile(path):
        print("文件不存在:", path)
        sys.exit(1)

    docs = pdf_loader(path, max_pages=0, log_progress_every=0)
    print("path:", path)
    print("documents:", len(docs))
    for i, d in enumerate(docs):
        pc = d.page_content or ""
        print(f"--- page meta={d.metadata} chars={len(pc)} ---")
        print(pc[:1200])
        if len(pc) > 1200:
            print("... [truncated]")


if __name__ == "__main__":
    main()
