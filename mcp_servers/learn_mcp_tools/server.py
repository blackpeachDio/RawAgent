"""
一个“最小可学习”的 MCP Server。

目标：
1) 给 Cursor/Agent 提供可调用的工具（tool calling）。
2) 展示 MCP server 如何暴露 tools、如何接收参数并返回结果。

建议：
- 只实现 stdio transport，和 Cursor 的本地启动方式最兼容。
- 通过 .cursor/mcp.json 把它接入 Cursor。
"""

import anyio
import click
from datetime import datetime
from zoneinfo import ZoneInfo

from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server


async def handle_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    # 告诉客户端（Cursor）有哪些工具可用、每个工具的入参 schema。
    return types.ListToolsResult(
        tools=[
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
    )


async def handle_call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    # 接收 Cursor 发来的“工具名 + 参数”，执行后把结果返回给 Cursor。
    args = params.arguments or {}

    if params.name == "get_current_time":
        tz = args.get("timezone")
        if tz:
            # ZoneInfo 让你可以指定 IANA 时区名（不需要额外依赖 pytz）。
            now = datetime.now(tz=ZoneInfo(tz))
        else:
            now = datetime.now()
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=now.isoformat(timespec="seconds"))]
        )

    if params.name == "echo":
        text = args.get("text", "")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(text))]
        )

    raise ValueError(f"Unknown tool: {params.name}")


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="MCP transport type。Cursor 本地建议使用 stdio。",
)
def main(transport: str) -> int:
    # Server 的第一个参数是服务名，会出现在调试日志/客户端展示中。
    app = Server(
        "rawagent-learn-mcp-tools",
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )

    if transport != "stdio":
        # 为了保持“最小示例”，这里先不实现 streamable-http。
        raise click.ClickException("This minimal example only supports --transport stdio")

    async def arun() -> None:
        # stdio_server 返回两条流：输入/输出；用来和 Cursor 进行进程级通信。
        async with stdio_server() as streams:
            await app.run(
                streams[0],
                streams[1],
                app.create_initialization_options(),
            )

    anyio.run(arun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

