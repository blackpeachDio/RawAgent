"""SSE（Server-Sent Events）帧编码与增量解析。"""
from __future__ import annotations


def format_sse_event(data: str, event: str | None = None) -> bytes:
    """
    编码一条 SSE 事件（以空行结束）。
    data 中若含换行，按规范拆成多行 data:。
    """
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for segment in data.split("\n"):
        lines.append(f"data: {segment}")
    lines.append("")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


class SSEDecoder:
    """从字节流中解析出完整 SSE 事件 (event_name, data_payload)。"""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[str | None, str]]:
        self._buf.extend(chunk)
        out: list[tuple[str | None, str]] = []
        while True:
            sep = self._buf.find(b"\n\n")
            if sep < 0:
                sep = self._buf.find(b"\r\n\r\n")
                crlf = True
            else:
                crlf = False
            if sep < 0:
                break
            raw = bytes(self._buf[:sep])
            del self._buf[: sep + (4 if crlf else 2)]
            event: str | None = None
            data_lines: list[str] = []
            for line in raw.splitlines():
                if line.startswith(b"event:"):
                    event = line[6:].strip().decode("utf-8", errors="replace") or None
                elif line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip().decode("utf-8", errors="replace"))
            payload = "\n".join(data_lines)
            out.append((event, payload))
        return out
