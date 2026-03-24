"""
一个“最小可学习”的 MCP Server。

目标：
1) 给 Cursor/Agent 提供可调用的工具（tool calling）。
2) 展示 MCP server 如何暴露 tools、如何接收参数并返回结果。

建议：
- 只实现 stdio transport，和 Cursor 的本地启动方式最兼容。
- 通过 .cursor/mcp.json 把它接入 Cursor。

说明：mcp>=1.3 使用装饰器注册 handler（`ServerRequestContext` 等旧 API 已移除）。
"""

import anyio
import click
from datetime import datetime
from zoneinfo import ZoneInfo

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_current_time",
            title="Get Current Time",
            description="返回当前时间字符串；可选 timezone。",
            input_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "例如 Asia/Shanghai。缺省则使用系统本地时区。",
                    }
                },
            },
        ),
        types.Tool(
            name="echo",
            title="Echo",
            description="原样返回输入文本（用于验证工具调用链路是否通畅）。",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要回显的文本",
                    }
                },
            },
        ),
    ]


def main() -> int:
    app = Server("rawagent-learn-mcp-tools")

    @app.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return _build_tools()

    @app.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        args = arguments or {}

        if name == "get_current_time":
            tz = args.get("timezone")
            if tz:
                now = datetime.now(tz=ZoneInfo(str(tz)))
            else:
                now = datetime.now()
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=now.isoformat(timespec="seconds"))]
            )

        if name == "echo":
            text = args.get("text", "")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(text))]
            )

        raise ValueError(f"Unknown tool: {name}")

    async def arun() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )

    anyio.run(arun)
    return 0


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="MCP transport type。Cursor 本地建议使用 stdio。",
)
def cli(transport: str) -> int:
    if transport != "stdio":
        raise click.ClickException("This minimal example only supports --transport stdio")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
