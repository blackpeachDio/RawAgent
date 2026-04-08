"""
配置加载：路径基于仓库根 ``config/``（``get_repo_root()``），与原先 ``get_abs_path("../config/...")`` 等价。

``get_config_path`` 定义在 ``utils.path_utils``，此处再导出以兼容 ``from utils.config_utils import get_config_path``。
"""
from __future__ import annotations

import json
import os

import yaml

from utils.path_utils import get_config_path


def load_rag_config(config_path: str | None = None, encoding: str = "utf-8"):
    path = config_path or get_config_path("rag.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str | None = None, encoding: str = "utf-8"):
    """加载 chroma.yml，并与 chroma_memory.yml 合并为同一 dict（键名不变，仍为 chroma_conf）。"""
    path = config_path or get_config_path("chroma.yml")
    with open(path, "r", encoding=encoding) as f:
        data = yaml.load(f, Loader=yaml.FullLoader) or {}
    if not isinstance(data, dict):
        data = {}
    mem_path = get_config_path("chroma_memory.yml")
    if os.path.isfile(mem_path):
        with open(mem_path, "r", encoding=encoding) as f:
            mem = yaml.load(f, Loader=yaml.FullLoader) or {}
        if isinstance(mem, dict):
            data.update(mem)
    return data


def load_prompts_config(config_path: str | None = None, encoding: str = "utf-8"):
    path = config_path or get_config_path("prompts.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str | None = None, encoding: str = "utf-8"):
    """加载 agent.yml，并与 memory.yml 合并为同一 dict（后者键覆盖前者，便于拆分文件而保持 agent_conf 接口不变）。"""
    path = config_path or get_config_path("agent.yml")
    with open(path, "r", encoding=encoding) as f:
        data = yaml.load(f, Loader=yaml.FullLoader) or {}
    if not isinstance(data, dict):
        data = {}
    memory_path = get_config_path("memory.yml")
    if os.path.isfile(memory_path):
        with open(memory_path, "r", encoding=encoding) as f:
            mem = yaml.load(f, Loader=yaml.FullLoader) or {}
        if isinstance(mem, dict):
            data.update(mem)
    return data


def load_skills_config(config_path: str | None = None, encoding: str = "utf-8"):
    """RawAgent 运行时技能（raw_agent_skills/）；文件不存在则关闭。"""
    path = config_path or get_config_path("skills.yml")
    if not os.path.isfile(path):
        return {"enabled": False, "root": "raw_agent_skills", "max_body_chars": 12000}
    with open(path, "r", encoding=encoding) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data if isinstance(data, dict) else {}


def load_mcp_config(config_path: str | None = None, encoding: str = "utf-8"):
    """Cursor 风格 MCP：见 config/mcp.json（mcpServers + transportType）。"""
    path = config_path or get_config_path("mcp.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def load_api_config(config_path: str | None = None, encoding: str = "utf-8"):
    """支持 config/api.yml；Docker/CI 可仅用环境变量 DASHSCOPE_API_KEY（会覆盖文件中的值）。"""
    path = config_path or get_config_path("api.yml")
    data: dict = {}
    if os.path.isfile(path):
        with open(path, "r", encoding=encoding) as f:
            data = yaml.load(f, Loader=yaml.FullLoader) or {}
    env_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if env_key:
        data["dashscope_api_key"] = env_key
    if not data.get("dashscope_api_key"):
        raise ValueError(
            "缺少 dashscope_api_key：请在 config/api.yml 中配置，或设置环境变量 DASHSCOPE_API_KEY。"
        )
    return data


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
skills_conf = load_skills_config()
mcp_conf = load_mcp_config()
api_conf = load_api_config()

# if __name__ == '__main__':
#     print(rag_conf["chat_model_name"])
