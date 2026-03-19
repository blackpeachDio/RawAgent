"""
提供统一的绝对路径
"""

import os


def get_project_root() -> str:
    """
    获取所在根目录
    :return: str 根目录
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(file_path)


def get_abs_path(relative_path: str) -> str:
    """
    相对路径
    :return: 绝对路径
    """
    project_path = get_project_root()
    return os.path.join(project_path, relative_path)


if __name__ == '__main__':
    print(get_abs_path(__file__))
    print(get_project_root())
