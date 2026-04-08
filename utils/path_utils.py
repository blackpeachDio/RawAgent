"""
提供统一的绝对路径。

- ``get_repo_root()``：仓库根目录（含 ``config/``、``agent/`` 等顶层包）。
- ``get_project_root()``：历史命名，实为 **本包所在目录**（即 ``utils/``），保留以兼容现有
  ``get_abs_path("../config/...")`` 等写法。
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


def get_abs_path(relative_path: str) -> str:
    """
    相对 ``utils/`` 的路径解析为绝对路径（与历史行为一致）。
    例如 ``../config/agent.yml`` → 仓库根下 ``config/agent.yml``。
    """
    return os.path.join(get_project_root(), relative_path)


if __name__ == "__main__":
    print("repo_root:", get_repo_root())
    print("utils (=get_project_root):", get_project_root())
    print("agent.yml:", get_abs_path("../config/agent.yml"))
