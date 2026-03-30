"""从配置中的远程 MCP（HTTP / SSE 等）加载 LangChain tools。配置：config/mcp.json（Cursor 风格 mcpServers）。"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.config_utils import mcp_conf
from utils.log_utils import logger


def _mcp_server_entries(mcp_cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容 Cursor 风格 `mcpServers` 与旧版扁平 `connections`。"""
    if mcp_cfg.get("mcpServers"):
        return mcp_cfg["mcpServers"]
    return mcp_cfg.get("connections") or {}


def _build_connections() -> dict[str, Any]:
    mcp_cfg = mcp_conf or {}
    raw = _mcp_server_entries(mcp_cfg)
    out: dict[str, Any] = {}
    for name, conn in raw.items():
        if not isinstance(conn, dict):
            continue
        if conn.get("disabled"):
            logger.info("[mcp] 跳过（disabled）: %s", name)
            continue
        url = (conn.get("url") or "").strip()
        if not url:
            logger.warning("[mcp] 跳过 %s：缺少 url", name)
            continue
        # Cursor: transportType；旧 YAML：transport
        tt = (conn.get("transportType") or conn.get("transport") or "http").strip().lower()
        headers = conn.get("headers")
        entry: dict[str, Any]

        if tt == "sse":
            entry = {"transport": "sse", "url": url}
        elif tt in ("http", "https", "streamable_http", "streamable-http"):
            entry = {"transport": "http", "url": url}
        else:
            logger.warning("[mcp] 跳过 %s：不支持的 transportType/transport=%s", name, tt)
            continue

        if isinstance(headers, dict) and headers:
            entry["headers"] = headers
        out[name] = entry
    return out


def load_remote_mcp_tools_sync() -> list[Any]:
    """
    同步加载远程 MCP 工具列表；失败行为由 config/mcp.json 中 strict 控制。
    """
    mcp_cfg = mcp_conf or {}
    if not mcp_cfg.get("enabled", False):
        return []
    connections = _build_connections()
    if not connections:
        logger.warning("[mcp] enabled=true 但未配置有效 mcpServers")
        return []

    strict = bool(mcp_cfg.get("strict", False))
    prefix = bool(mcp_cfg.get("tool_name_prefix", True))
    client = MultiServerMCPClient(connections, tool_name_prefix=prefix)

    async def _load():
        return await client.get_tools()

    try:
        tools = asyncio.run(_load())
        logger.info("[mcp] 已加载远程工具 %s 个: %s", len(tools), [getattr(t, "name", str(t)) for t in tools])
        return list(tools)
    except Exception as e:
        msg = f"[mcp] 连接远程 MCP 失败: {e}"
        if strict:
            logger.error(msg)
            raise RuntimeError(msg) from e
        logger.warning("%s；已跳过 MCP 工具，仅使用本地 tools", msg)
        return []
