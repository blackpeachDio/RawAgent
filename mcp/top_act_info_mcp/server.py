"""
top_act_info_mcp：通过 Streamable HTTP 暴露 MCP 工具 get_top_act_by_id（内部以 HTTP POST 转发占位域名）。

Agent 侧连接见 config/mcp.json（mcpServers.top_act_info_mcp）。

启动（项目根目录）:
  python mcp/top_act_info_mcp/server.py

默认监听: http://127.0.0.1:8010/mcp

环境变量 TOP_ACT_API_URL：覆盖默认占位 POST 地址（对方 @RequestBody Integer id，body 为 JSON 数字）。
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastmcp import FastMCP

mcp = FastMCP("top_act_info_mcp")

# 占位域名与路径，部署时改为真实地址或通过环境变量 TOP_ACT_API_URL 覆盖
_DEFAULT_POST_URL = "https://placeholder.example.com/api/top-activity/by-id"


@mcp.tool()
def get_top_act_by_id(id: int) -> str:
    """
    根据活动 ID 查询置顶活动信息：向远端 POST application/json，body 为 JSON 数字 id（对应 Spring @RequestBody Integer id）。
    """
    url = (os.environ.get("TOP_ACT_API_URL") or "").strip() or _DEFAULT_POST_URL
    payload = json.dumps(id).encode("utf-8")
    req = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "top_act_info_mcp/1.0",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return body
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return json.dumps(
            {"error": "http", "status": e.code, "detail": raw},
            ensure_ascii=False,
        )
    except URLError as e:
        return json.dumps(
            {"error": "network", "detail": str(e.reason)},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8010,
        path="/mcp",
    )
