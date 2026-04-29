"""在独立线程中启动内嵌 uvicorn（供 Streamlit 单进程联调）。"""
from __future__ import annotations

import os
import threading
import time

_embed_lock = threading.Lock()
_embed_started = False


def ensure_embedded_sse_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """幂等：启动一次内嵌 SSE 服务。"""
    global _embed_started
    if os.environ.get("RAW_AGENT_EMBED_SSE", "1").strip().lower() in ("0", "false", "no"):
        return
    with _embed_lock:
        if _embed_started:
            return
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "内嵌 SSE 需要安装 uvicorn：pip install uvicorn[standard]"
            ) from e

        from api.sse_chat import app as sse_app

        def _run() -> None:
            uvicorn.run(
                sse_app,
                host=host,
                port=port,
                log_level=os.environ.get("RAW_AGENT_SSE_LOG_LEVEL", "warning"),
            )

        t = threading.Thread(target=_run, name="raw-agent-sse-uvicorn", daemon=True)
        t.start()
        deadline = time.time() + 8.0
        ok = False
        while time.time() < deadline:
            try:
                import http.client

                c = http.client.HTTPConnection(host, port, timeout=1.0)
                c.request("GET", "/health")
                r = c.getresponse()
                body = r.read()
                c.close()
                if r.status == 200 and body:
                    ok = True
                    break
            except OSError:
                time.sleep(0.05)
        if not ok:
            raise RuntimeError(
                f"内嵌 SSE 在 8s 内未就绪（{host}:{port}）。请检查端口占用或改用独立 uvicorn。"
            )
        _embed_started = True
