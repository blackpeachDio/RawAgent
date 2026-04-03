"""
top_act_info_mcp：通过 Streamable HTTP 暴露 MCP 工具 get_top_act_by_id（内部以 HTTP POST 转发占位域名）。

Agent 侧连接见 config/mcp.json（mcpServers.top_act_info_mcp）。

启动（项目根目录）:
  python mcp_servers/top_act_info_mcp/server.py

默认监听: http://127.0.0.1:8010/mcp

环境变量：
  TOP_ACT_API_URL：覆盖默认 POST 地址（对方 @RequestBody Integer id）。
  TOP_ACT_COOKIE：整段 Cookie 字符串（请求头 Cookie）。
  TOP_ACT_COOKIE_FILE：Cookie 文件路径（与 TOP_ACT_COOKIE 二选一，优先环境变量）。
  若存在项目根 config/top_act_cookie.txt（已 gitignore），也会自动读取。
"""
from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastmcp import FastMCP

mcp = FastMCP("top_act_info_mcp")
_logger = logging.getLogger("top_act_info_mcp")

# 占位域名与路径，部署时改为真实地址或通过环境变量 TOP_ACT_API_URL 覆盖
_DEFAULT_POST_URL = "http://11.63.160.19/mvc/news/activity/getActivityById.do"


def _cookie_header() -> str | None:
    """从环境变量、TOP_ACT_COOKIE_FILE 或 config/top_act_cookie.txt 读取 Cookie。"""
    raw = (os.environ.get("TOP_ACT_COOKIE") or "").strip()
    if raw:
        return raw
    path = (os.environ.get("TOP_ACT_COOKIE_FILE") or "").strip()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    here = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.normpath(os.path.join(here, "..", "..", "config", "top_act_cookie.txt"))
    if os.path.isfile(default_file):
        with open(default_file, encoding="utf-8") as f:
            return f.read().strip() or None
    return None


@mcp.tool()
def get_top_act_by_id(id: int) -> str:
    """
    根据活动 ID 查询置顶活动信息：向远端 POST application/json，body 为 JSON 数字 id（对应 Spring @RequestBody Integer id）。
    """
    url = (os.environ.get("TOP_ACT_API_URL") or "").strip() or _DEFAULT_POST_URL
    payload = json.dumps(id).encode("utf-8")
    _logger.info("get_top_act_by_id 调用 id=%s url=%s", id, url)
    print(f"[top_act_info_mcp] 调用 get_top_act_by_id(id={id}) POST {url}", flush=True)
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "top_act_info_mcp/1.0",
    }
    cookie = _cookie_header()
    if cookie:
        headers["Cookie"] = cookie
        _logger.info("请求已附带 Cookie（长度=%s）", len(cookie))
    else:
        _logger.warning("未配置 Cookie（TOP_ACT_COOKIE / TOP_ACT_COOKIE_FILE / config/top_act_cookie.txt），若接口需登录可能失败")
    parsed = urlparse(url)
    if parsed.netloc:
        headers["Host"] = parsed.netloc
    headers["Content-Length"] = str(len(payload))
    req = Request(
        url,
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", None)
            _logger.info(
                "get_top_act_by_id 成功 id=%s status=%s len=%s body=%s",
                id,
                status,
                len(body),
                body if len(body) <= 4000 else body[:4000] + "...(truncated)",
            )
            print(
                f"[top_act_info_mcp] 结果 id={id} status={status} len={len(body)}\n"
                f"{body if len(body) <= 2000 else body[:2000] + '...(truncated)'}",
                flush=True,
            )
            return body
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        err = json.dumps(
            {"error": "http", "status": e.code, "detail": raw},
            ensure_ascii=False,
        )
        _logger.warning("get_top_act_by_id HTTPError id=%s: %s", id, err)
        print(f"[top_act_info_mcp] HTTPError id={id} {err}", flush=True)
        return err
    except URLError as e:
        err = json.dumps(
            {"error": "network", "detail": str(e.reason)},
            ensure_ascii=False,
        )
        _logger.warning("get_top_act_by_id URLError id=%s: %s", id, err)
        print(f"[top_act_info_mcp] URLError id={id} {err}", flush=True)
        return err


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8010,
        path="/mcp",
    )
