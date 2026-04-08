"""供 create_agent 注册的技能工具（依赖 loader，与 ``tools.agent_tools`` 内建工具隔离）。"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from raw_agent_skillkit.loader import list_skill_entries, read_skill_body


@tool
def list_agent_skills() -> str:
    """列出当前可用的 RawAgent 技能 id、名称与简介（来自各目录下 SKILL.md 的 frontmatter）。禁用或目录为空时返回说明字符串。"""
    rows = list_skill_entries()
    if not rows:
        return "当前无可用技能（检查 config/skills.yml 中 enabled 与 root 目录，或 raw_agent_skills/*/SKILL.md）。"
    lines = []
    for r in rows:
        lines.append(
            json.dumps(
                {"id": r["id"], "name": r["name"], "description": r["description"]},
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


@tool
def get_agent_skill(skill_id: str) -> str:
    """按技能 id（子目录名）读取 SKILL.md 正文（不含 frontmatter），用于按需加载长说明，避免占满工具定义 token。skill_id 须与 list_agent_skills 中一致。"""
    got = read_skill_body(skill_id.strip())
    if got is None:
        return f"未找到技能：{skill_id!r}（请先用 list_agent_skills 查看 id）。"
    _meta, body = got
    return body


def build_skill_tools():
    """enabled 时返回工具列表，否则空列表。"""
    from utils.config_utils import skills_conf

    if not skills_conf.get("enabled", False):
        return []
    return [list_agent_skills, get_agent_skill]
