"""
user_info_mcp：通过 Streamable HTTP 暴露 MCP 工具（仅远程连接，本进程不供 Agent 直接 import 调用）。
Agent 侧连接列表见 config/mcp.json（mcpServers.user_info_mcp）。

启动（项目根目录）:
  python mcp/user_info_mcp/server.py

默认监听: http://127.0.0.1:8010/mcp
"""
from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("user_info_mcp")

# 演示用内存库；可替换为真实用户服务
_MOCK_USERS: dict[str, dict[str, str]] = {
    "123": {"user_id": "123", "name": "张三", "email": "zhangsan@example.com", "tier": "vip"},
    "1001": {"user_id": "1001", "name": "演示用户1001", "email": "u1001@example.com", "tier": "standard"},
}


@mcp.tool()
def get_user_info_by_id(user_id: str) -> str:
    """根据用户 ID 查询用户信息，返回 JSON 字符串；无记录时返回说明信息。"""
    import json

    uid = (user_id or "").strip()
    if not uid:
        return json.dumps({"error": "user_id 不能为空"}, ensure_ascii=False)
    row = _MOCK_USERS.get(uid)
    if row is None:
        return json.dumps(
            {"user_id": uid, "found": False, "message": "未找到该用户（演示数据仅含 123、1001 等）"},
            ensure_ascii=False,
        )
    return json.dumps({"found": True, **row}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8010,
        path="/mcp",
    )
