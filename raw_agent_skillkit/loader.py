"""从 config/skills.yml 与 raw_agent_skills/ 加载 SKILL.md（与业务代码隔离）。"""
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
