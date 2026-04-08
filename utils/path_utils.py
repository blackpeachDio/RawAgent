"""
提供统一的绝对路径。

- ``get_repo_root()``：仓库根目录（含 ``config/``、``agent/`` 等顶层包）。
- ``get_config_path(name)``：``config/<name>`` 的绝对路径（不加载 YAML）。
- ``resolve_repo_path()``：配置项里 ``../...`` 或仓库根相对路径。
- ``get_project_root()``：历史命名，实为 **本包所在目录**（即 ``utils/``），保留以兼容
  旧版 ``get_abs_path`` 中非 ``../`` 的相对路径。
"""

import os

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_repo_root() -> str:
    """仓库根目录（``utils`` 的父目录）。"""
    return os.path.abspath(os.path.join(_UTILS_DIR, ".."))


def get_project_root() -> str:
    """
    含 ``path_utils`` 的包目录（当前为 ``utils/``）。

    历史原因命名为 project_root；新代码请用 ``get_repo_root()`` 表示仓库根。
    """
    return os.path.abspath(_UTILS_DIR)


def get_config_path(filename: str) -> str:
    """仓库根下 ``config/<filename>`` 的绝对路径（仅路径拼接，不读文件）。"""
    return os.path.normpath(os.path.join(get_repo_root(), "config", filename))


def resolve_repo_path(relative_path: str) -> str:
    """
    将配置里的相对路径解析为绝对路径（**相对仓库根**）。

    配置中常见 ``../xxx``（历史上与 ``get_abs_path`` 相对 ``utils/`` 等价）→ ``{repo}/xxx``。
    不以 ``../`` 开头的相对路径则视为相对仓库根，例如 ``data/foo`` → ``{repo}/data/foo``。
    已是绝对路径则原样规范化后返回。
    """
    p = (relative_path or "").strip()
    if not p:
        return p
    if os.path.isabs(p):
        return os.path.normpath(p)
    if p.startswith("../"):
        rest = p[3:].lstrip("/\\")
        return os.path.normpath(os.path.join(get_repo_root(), rest))
    return os.path.normpath(os.path.join(get_repo_root(), p))


def get_abs_path(relative_path: str) -> str:
    """
    相对 ``utils/`` 的路径解析为绝对路径（与历史行为一致）。新代码请用
    ``resolve_repo_path``（仓库根语义）或 ``get_config_path``（仅 ``config/`` 下文件）。

    以 ``../`` 开头时与 ``resolve_repo_path`` 结果相同；否则为 ``utils/<relative_path>``。
    """
    p = (relative_path or "").strip()
    if p.startswith("../"):
        return resolve_repo_path(p)
    return os.path.normpath(os.path.join(get_project_root(), p))


if __name__ == "__main__":
    print("repo_root:", get_repo_root())
    print("utils (=get_project_root):", get_project_root())
    print("agent.yml (legacy get_abs_path):", get_abs_path("../config/agent.yml"))
    print("agent.yml (get_config_path):", get_config_path("agent.yml"))
    print("resolve ../history:", resolve_repo_path("../history"))
