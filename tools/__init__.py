"""
内建 Agent 工具与中间件的稳定包名（实现仍在 ``agent.tools``）。

后续若迁移工具实现，可在此包内 re-export，保持 ``from agent.tools.xxx`` 或
``agent.tools`` 的 import 路径由 ``agent/tools`` 转发。
"""

__all__: list[str] = []
