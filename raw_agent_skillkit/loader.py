"""从 config/skills.yml 与 raw_agent_skills/ 加载 SKILL.md，拼入 system。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from utils.config_utils import skills_conf
from utils.path_utils import resolve_repo_path


def skills_root_abs() -> str:
    rel = (skills_conf.get("root") or "raw_agent_skills").strip()
    if os.path.isabs(rel):
        return rel
    return resolve_repo_path(os.path.join("..", rel))


def _parse_skill_md(text: str) -> tuple[dict, str]:
    """解析可选 YAML frontmatter，返回 (meta, body)。"""
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except Exception:
        meta = {}
    body = (parts[2] or "").strip()
    return (meta if isinstance(meta, dict) else {}), body


def list_skill_entries() -> list[dict]:
    """[{id, name, description}]，id 为子目录名。"""
    root = skills_root_abs()
    if not os.path.isdir(root):
        return []
    out: list[dict] = []
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or "__" in name:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        skill_file = os.path.join(path, "SKILL.md")
        if not os.path.isfile(skill_file):
            continue
        try:
            with open(skill_file, encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            continue
        meta, _ = _parse_skill_md(raw)
        out.append(
            {
                "id": name,
                "name": str(meta.get("name") or name),
                "description": str(meta.get("description") or "").strip(),
            }
        )
    return out


def read_skill_body(skill_id: str) -> tuple[dict, str] | None:
    """返回 (meta, body)；skill_id 非法或不存在则 None。"""
    sid = (skill_id or "").strip()
    if not sid or ".." in sid or "/" in sid or "\\" in sid:
        return None
    root = Path(skills_root_abs()).resolve()
    try:
        skill_file = (root / sid / "SKILL.md").resolve()
        skill_file.relative_to(root)
    except ValueError:
        return None
    if not skill_file.is_file():
        return None
    try:
        with open(skill_file, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return None
    meta, body = _parse_skill_md(raw)
    max_chars = int(skills_conf.get("max_body_chars") or 12000)
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[已截断，max_body_chars={max_chars}]"
    return meta, body


_inject_system_block_cache: str | None = None


def build_skills_system_inject_block() -> str:
    """
    将各技能 SKILL.md 正文拼成一段，供 middleware 追加到 system。
    进程内缓存；改 SKILL.md 后需重启进程生效。
    """
    global _inject_system_block_cache
    if not skills_conf.get("enabled", False):
        return ""
    if _inject_system_block_cache is not None:
        return _inject_system_block_cache

    entries = list_skill_entries()
    if not entries:
        _inject_system_block_cache = ""
        return ""

    raw_tot = skills_conf.get("inject_max_total_chars", 24000)
    total_max = int(raw_tot) if raw_tot is not None else 24000

    parts: list[str] = [
        "\n\n---\n## 已注入技能\n下列说明已常驻上下文中，按其中流程直接调用相应工具即可。\n",
    ]
    used = 0
    for e in entries:
        sid = e["id"]
        name = str(e.get("name") or sid)
        got = read_skill_body(sid)
        if not got:
            continue
        _meta, body = got
        header = f"\n### 技能 [{sid}] {name}\n\n"
        block = header + body
        if total_max > 0 and used + len(block) > total_max:
            room = total_max - used - len(header)
            if room < 120:
                parts.append("\n\n[后续技能因 inject_max_total_chars 上限已省略]\n")
                break
            block = header + body[: max(0, room)] + "\n\n[本技能正文已按总预算截断]\n"
        parts.append(block)
        used += len(block)

    _inject_system_block_cache = "".join(parts)
    return _inject_system_block_cache
