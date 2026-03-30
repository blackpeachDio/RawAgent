"""从配置中的远程 MCP（HTTP / SSE 等）加载 LangChain tools。配置：config/mcp.json（Cursor 风格 mcpServers）。"""
from __future__ import annotations

import asyncio
import concurrent.futures
import traceback
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.config_utils import mcp_conf
from utils.log_utils import logger


def _run_coro_sync(coro: Any) -> Any:
    """在同步上下文中执行协程（兼容已在 asyncio 循环内的情况）。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(asyncio.run, coro)
        return fut.result()


def _ensure_sync_mcp_tool(tool: Any) -> Any:
    """
    langchain-mcp-adapters 常生成仅有 coroutine、无 func 的 StructuredTool；
    create_agent 的工具节点走同步 invoke，会触发「StructuredTool does not support sync invocation」。
    包一层同步 func，内部 ainvoke。
    """
    if getattr(tool, "func", None) is not None:
        return tool
    if getattr(tool, "coroutine", None) is None:
        return tool

    name = getattr(tool, "name", "mcp_tool")
    description = getattr(tool, "description", "") or ""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        logger.warning("[mcp] 工具 %s 缺少 args_schema，无法包同步包装", name)
        return tool

    return_direct = bool(getattr(tool, "return_direct", False))
    response_format = getattr(tool, "response_format", "content")

    def _sync(**kwargs: Any) -> Any:
        return _run_coro_sync(tool.ainvoke(kwargs))

    try:
        wrapped = StructuredTool.from_function(
            func=_sync,
            name=name,
            description=description,
            args_schema=args_schema,
            infer_schema=False,
            return_direct=return_direct,
            response_format=response_format,
        )
        logger.debug("[mcp] 已为工具 %s 添加同步 func 包装", name)
        return wrapped
    except Exception as e:
        logger.warning("[mcp] 工具 %s 同步包装失败，保持原工具: %s", name, e)
        return tool


def _log_wrapped_exception(exc: BaseException, prefix: str = "[mcp]") -> None:
    """展开 TaskGroup / ExceptionGroup 等包装，避免只看到一句 unhandled errors in a TaskGroup。"""
    logger.error("%s 顶层: %s: %s", prefix, type(exc).__name__, exc)
    if exc.__cause__ is not None:
        logger.error("%s __cause__: %s: %s", prefix, type(exc.__cause__).__name__, exc.__cause__)
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        logger.error("%s __context__: %s: %s", prefix, type(exc.__context__).__name__, exc.__context__)
    subs = getattr(exc, "exceptions", None)
    if subs:
        for i, sub in enumerate(subs):
            logger.error("%s 子异常[%s]: %s: %s", prefix, i, type(sub).__name__, sub)
            tb = getattr(sub, "__traceback__", None)
            if tb is not None:
                logger.error(
                    "%s 子异常[%s] traceback:\n%s",
                    prefix,
                    i,
                    "".join(traceback.format_exception(type(sub), sub, tb)),
                )


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


async def _load_tools_per_server(
        connections: dict[str, Any],
        *,
        tool_name_prefix: bool,
        strict: bool,
) -> list[Any]:
    """
    按服务器分别连接。避免 MultiServerMCPClient 一次连多台时 TaskGroup 只报一条笼统错误，
    无法判断是哪台失败。
    """
    all_tools: list[Any] = []
    for name, conn_cfg in connections.items():
        try:
            client = MultiServerMCPClient({name: conn_cfg}, tool_name_prefix=tool_name_prefix)
            tools = [_ensure_sync_mcp_tool(t) for t in await client.get_tools()]
            names = [getattr(t, "name", str(t)) for t in tools]
            logger.info("[mcp] 服务器 %s 成功，工具 %s 个: %s", name, len(tools), names)
            all_tools.extend(tools)
        except Exception as e:
            logger.error(
                "[mcp] 服务器 %s 连接失败（url=%s transport=%s）",
                name,
                conn_cfg.get("url"),
                conn_cfg.get("transport"),
            )
            _log_wrapped_exception(e, prefix=f"[mcp:{name}]")
            if strict:
                raise RuntimeError(f"[mcp] 服务器 {name} 连接失败: {e}") from e
    return all_tools


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

    async def _run():
        return await _load_tools_per_server(connections, tool_name_prefix=prefix, strict=strict)

    try:
        tools = asyncio.run(_run())
        if not tools:
            logger.warning("[mcp] 未加载到任何远程工具（请查看上方各服务器错误日志）")
        else:
            logger.info(
                "[mcp] 合计远程工具 %s 个: %s",
                len(tools),
                [getattr(t, "name", str(t)) for t in tools],
            )
        return list(tools)
    except Exception as e:
        msg = f"[mcp] 连接远程 MCP 失败: {e}"
        _log_wrapped_exception(e)
        if strict:
            logger.error(msg)
            raise RuntimeError(msg) from e
        logger.warning("%s；已跳过 MCP 工具，仅使用本地 tools", msg)
        return []
